from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import models, transforms


class Places365SceneClassifier:
    """Classifies images as indoor or outdoor using Places365 ResNet-50."""

    def __init__(
        self,
        model_path: Path,
        categories_path: Path,
        io_path: Path,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_path = Path(model_path)
        self.categories_path = Path(categories_path)
        self.io_path = Path(io_path)
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.categories = self._load_categories()
        self.indoor_idxs, self.outdoor_idxs = self._load_io_split()
        self.transform = self._build_transforms()
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.categories))
        state_dict = torch.load(self.model_path, map_location="cpu")
        # Original checkpoint uses key "state_dict".
        if "state_dict" in state_dict:
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict["state_dict"].items()
            }
        model.load_state_dict(state_dict, strict=True)
        return model

    def _load_categories(self) -> List[str]:
        categories: List[str] = []
        with self.categories_path.open("r", encoding="utf-8") as f:
            for line in f:
                category = line.strip().split(" ", maxsplit=1)[0]
                categories.append(category.strip())
        return categories

    def _load_io_split(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indoor = []
        outdoor = []
        with self.io_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                _, flag = line.strip().rsplit(" ", maxsplit=1)
                if flag == "1":  # indoor
                    indoor.append(idx)
                elif flag == "2":  # outdoor
                    outdoor.append(idx)
                else:
                    raise ValueError(f"Unexpected IO flag '{flag}' on line {idx}")
        return (
            torch.tensor(indoor, dtype=torch.long),
            torch.tensor(outdoor, dtype=torch.long),
        )

    def _build_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def classify_paths(
        self,
        image_paths: Sequence[Path],
        threshold: float = 0.1,
    ) -> List[dict]:
        """Return label (indoor/outdoor/unknown) and confidences for image paths."""
        results: List[dict] = []
        with torch.no_grad():
            for batch_paths in self._batch(image_paths, self.batch_size):
                images = [self._load_image(path) for path in batch_paths]
                tensor_batch = torch.stack(images).to(self.device)
                logits = self.model(tensor_batch)
                probs = F.softmax(logits, dim=1).cpu()

                indoor_scores = probs[:, self.indoor_idxs].sum(dim=1)
                outdoor_scores = probs[:, self.outdoor_idxs].sum(dim=1)

                for path, indoor_score, outdoor_score in zip(
                    batch_paths, indoor_scores.tolist(), outdoor_scores.tolist()
                ):
                    label, confidence = self._decide_label(
                        indoor_score, outdoor_score, threshold
                    )
                    results.append(
                        {
                            "filename": str(path),
                            "scene_type": label,
                            "indoor_confidence": indoor_score,
                            "outdoor_confidence": outdoor_score,
                            "confidence": confidence,
                        }
                    )
        return results

    def _decide_label(
        self, indoor_score: float, outdoor_score: float, threshold: float
    ) -> Tuple[str, float]:
        diff = abs(indoor_score - outdoor_score)
        if diff < threshold:
            return "unknown", diff
        if indoor_score > outdoor_score:
            return "indoor", indoor_score
        return "outdoor", outdoor_score

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))

    @staticmethod
    def _batch(sequence: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
        batch: List[Path] = []
        for item in sequence:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


