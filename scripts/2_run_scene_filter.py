import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocessing.scene_classifier import Places365SceneClassifier


def collect_images(root: Path, recursive: bool = True) -> List[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    paths = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify images as indoor/outdoor using Places365 ResNet-50."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/images_by_continent"),
        help="Directory containing images to classify.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scene_predictions.csv"),
        help="Path to output CSV with predictions.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/places365/resnet50_places365.pth"),
        help="Path to Places365 ResNet-50 checkpoint.",
    )
    parser.add_argument(
        "--categories-path",
        type=Path,
        default=Path("models/places365/categories_places365.txt"),
        help="Path to categories file.",
    )
    parser.add_argument(
        "--io-path",
        type=Path,
        default=Path("models/places365/IO_places365.txt"),
        help="Path to indoor/outdoor mapping file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum difference between indoor/outdoor scores to assign a label.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (cuda/cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help=(
            "Make filenames in the CSV relative to this directory. "
            "Defaults to --image-dir if not provided."
        ),
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("coords_with_continent.csv"),
        help=(
            "Optional CSV with columns 'filename' and 'continent' to map each image "
            "to its continent."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search for images.",
    )
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    if args.mapping_csv and not args.mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping_csv}")

    image_paths = collect_images(args.image_dir, recursive=not args.no_recursive)
    if not image_paths:
        print("No images found to classify; exiting.")
        return

    classifier = Places365SceneClassifier(
        model_path=args.model_path,
        categories_path=args.categories_path,
        io_path=args.io_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    results = []
    for chunk in tqdm(
        [
            image_paths[i : i + classifier.batch_size]
            for i in range(0, len(image_paths), classifier.batch_size)
        ],
        total=(len(image_paths) + classifier.batch_size - 1) // classifier.batch_size,
        desc="Classifying scenes",
    ):
        results.extend(classifier.classify_paths(chunk, threshold=args.threshold))

    df = pd.DataFrame(results)
    base_dir = args.relative_to.resolve() if args.relative_to else args.image_dir.resolve()

    def to_relative(path_str: str) -> str:
        resolved = Path(path_str).resolve()
        try:
            return str(resolved.relative_to(base_dir))
        except ValueError:
            return resolved.name

    df["filename"] = df["filename"].apply(to_relative)

    df["continent"] = _assign_continents(
        df["filename"], args.mapping_csv, base_dir=base_dir
    )
    before_filter = len(df)
    df = df[df["scene_type"] != "unknown"].copy()
    dropped = before_filter - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} images labelled as 'unknown'.")

    df = df[["filename", "scene_type", "confidence", "continent"]]
    df.to_csv(args.output, index=False)
    print(f"Saved predictions for {len(df)} images to {args.output}")


def _assign_continents(
    filenames: pd.Series,
    mapping_csv: Optional[Path],
    base_dir: Path,
) -> pd.Series:
    if mapping_csv:
        mapping_df = pd.read_csv(mapping_csv)
        if "filename" not in mapping_df.columns or "continent" not in mapping_df.columns:
            raise ValueError(
                "Mapping CSV must contain columns 'filename' and 'continent'."
            )
        mapping = mapping_df.set_index("filename")["continent"]

        def map_from_csv(name: str) -> Optional[str]:
            candidates = [name, Path(name).name]
            for candidate in candidates:
                if candidate in mapping:
                    return mapping[candidate]
            return None

        continents = filenames.apply(map_from_csv)
    else:
        known_continents = {
            "Africa",
            "Americas",
            "Asia",
            "Europe",
            "Oceania",
            "North America",
            "South America",
            "Antarctica",
        }

        def infer_from_path(name: str) -> Optional[str]:
            parts = Path(name).parts
            for part in parts:
                if part in known_continents:
                    return part
            return None

        continents = filenames.apply(infer_from_path)

    if continents.isnull().any():
        missing = filenames[continents.isnull()]
        raise ValueError(
            "Could not determine continent for some files. "
            "Provide --mapping-csv or check directory structure. "
            f"Examples: {missing.head().tolist()}"
        )

    return continents


if __name__ == "__main__":
    main()

