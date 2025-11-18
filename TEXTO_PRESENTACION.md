# GeoVision: Clasificación Geográfica con Vision Transformer
## Texto de Presentación del Proyecto

---

Buenos días/tardes. Hoy les presentaré **GeoVision**, un proyecto de clasificación de imágenes geográficas que utiliza inteligencia artificial para identificar el continente de origen de una fotografía con una precisión del 75 por ciento.

## 1. Introducción y Motivación

¿Se han preguntado alguna vez cómo identificar la ubicación geográfica de una imagen solo con su contenido visual? Esta es una tarea compleja que combina visión por computadora con conocimiento geográfico. 

GeoVision aborda este desafío mediante el uso de técnicas avanzadas de deep learning para clasificar imágenes en cinco continentes: África, América, Asia, Europa y Oceanía. El proyecto es especialmente relevante porque puede aplicarse a múltiples contextos: desde la verificación de contenido geográfico en redes sociales, hasta la organización automática de archivos fotográficos personales o profesionales.

## 2. Objetivo del Proyecto

El objetivo principal de GeoVision es desarrollar un sistema de clasificación geográfica que sea capaz de identificar el continente de una imagen con alta precisión. Para lograrlo, el proyecto integra tres fuentes de información diferentes: características visuales extraídas de la imagen, embeddings semánticos de CLIP, y features de geolocalización basados en coordenadas GPS cuando están disponibles.

## 3. Tecnologías Utilizadas y Justificación

### 3.1 Vision Transformer (ViT)

Utilizamos **Vision Transformer**, específicamente el modelo `google/vit-base-patch16-224-in21k`, como backbone principal del sistema. Elegimos ViT porque:

- **Transformadores en visión**: Los transformers han demostrado ser superiores a las CNNs tradicionales en muchas tareas de visión por computadora
- **Pre-entrenamiento en ImageNet-21k**: El modelo viene pre-entrenado en 21 mil clases de ImageNet, lo que le proporciona una comprensión visual robusta
- **Arquitectura escalable**: Los transformers son más escalables y permiten mejor integración con otros componentes como CLIP

### 3.2 CLIP Fusionado

Integramos **CLIP (Contrastive Language-Image Pre-training)** de OpenAI, específicamente `openai/clip-vit-base-patch32`, mediante una estrategia de **fusión end-to-end**. Elegimos CLIP porque:

- **Representaciones semánticas**: CLIP fue entrenado con millones de pares imagen-texto, lo que le permite capturar información semántica más rica que los modelos puramente visuales
- **Mejora significativa**: La integración de CLIP aumenta el F1-score en aproximadamente 5 puntos porcentuales, de 70% a 75%
- **Fusión directa**: Integramos CLIP directamente en el modelo durante el entrenamiento, proyectando los embeddings de 512 dimensiones a 128 mediante una capa lineal con normalización y activación GELU

### 3.3 Features de Geolocalización

Añadimos **embeddings geográficos** basados en coordenadas GPS discretizadas en bins de 10 grados. Esta decisión se justifica porque:

- **Contexto geográfico**: Las coordenadas GPS proporcionan información contextual valiosa que complementa las características visuales
- **Discretización inteligente**: Dividimos la latitud en 18 bins de 10 grados cada uno, y la longitud en 36 bins, permitiendo que el modelo aprenda patrones geográficos sin ser demasiado específico
- **Robustez**: El modelo puede manejar coordenadas faltantes mediante un bin especial "unknown"

### 3.4 Places365 para Filtrado

Utilizamos **ResNet-50 pre-entrenado en Places365** para clasificar imágenes como indoor u outdoor. Esto es crucial porque:

- **Mejora la calidad del dataset**: Las imágenes indoor tienen mucho ruido para clasificación geográfica y reducen significativamente el rendimiento
- **Filtrado por confianza**: Solo conservamos imágenes con confianza superior al 60%, eliminando casos ambiguos que podrían confundir al modelo
- **Resultado**: Al entrenar solo con imágenes outdoor, el F1-score aumenta de 52% (con ambos) a 75% (solo outdoor)

### 3.5 PyTorch y Hugging Face

Elegimos **PyTorch** como framework principal y **Hugging Face Transformers** porque:

- **Facilidad de uso**: Hugging Face proporciona implementaciones optimizadas y pre-entrenadas de ViT y CLIP
- **Integración**: Permite combinar fácilmente múltiples modelos pre-entrenados en un solo pipeline
- **Reproducibilidad**: Facilita la reproducción de resultados y el compartir modelos

### 3.6 Otras Tecnologías Clave

- **Albumentations**: Para data augmentation agresiva que ayuda a prevenir overfitting
- **GeoPandas y Cartopy**: Para mapeo de coordenadas GPS a continentes usando shapefiles geoespaciales
- **WeightedRandomSampler**: Para balancear clases y compensar el desbalance del dataset

## 4. Arquitectura del Modelo

Nuestro modelo, llamado `GeoViTForImageClassification`, fusiona tres fuentes de información de manera inteligente:

### 4.1 Pipeline de Procesamiento

La imagen primero pasa por el **backbone ViT**, que la divide en patches de 16x16 píxeles, los procesa mediante atención multi-cabeza, y extrae un vector de características de 768 dimensiones.

Simultáneamente, si están disponibles las coordenadas GPS, estas se **discretizan en bins** de latitud y longitud. Cada bin se convierte en un embedding de 32 dimensiones mediante capas de embedding aprendidas.

Opcionalmente, si CLIP está habilitado, la imagen también se procesa con el **modelo CLIP**, generando un embedding semántico de 512 dimensiones que luego se proyecta a 128 dimensiones mediante una capa lineal con normalización y activación.

Finalmente, estos tres vectores - **768 del ViT, 64 de la geolocalización (32+32), y 128 de CLIP** - se **concatenan** en un vector de 960 dimensiones, pasan por dropout para regularización, y se alimentan a un clasificador lineal que produce las probabilidades para las 5 clases de continentes.

### 4.2 Decisiones de Diseño

- **Fusión temprana**: Decidimos concatenar las características tempranamente en lugar de usar un ensemble, porque esto permite al modelo aprender interacciones entre las diferentes fuentes de información durante el entrenamiento end-to-end
- **Dropout adicional**: Añadimos un dropout de 0.3 en el clasificador para reducir overfitting, especialmente importante dado el desbalance de clases

## 5. Pipeline de Procesamiento de Datos

El proyecto incluye un pipeline completo desde la extracción de datos hasta la predicción:

### 5.1 Extracción y Organización

1. **Extracción de imágenes**: Extraemos imágenes desde archivos MessagePack (formato binario eficiente) del dataset de Kaggle, generando un CSV con coordenadas GPS
2. **Mapeo geográfico**: Usando GeoPandas y Cartopy, mapeamos cada coordenada GPS a su continente correspondiente mediante spatial join con shapefiles
3. **Organización por continente**: Las imágenes se organizan en carpetas según su continente identificado

### 5.2 Filtrado y Preparación

4. **Clasificación indoor/outdoor**: Utilizamos Places365 para clasificar cada imagen y filtrar aquellas con baja confianza o que sean ambiguas
5. **Preparación de splits**: Generamos splits persistentes de entrenamiento y prueba (80-20) organizados por escena y continente, asegurando que no haya data leakage
6. **Cálculo de bins geográficos**: Para cada imagen calculamos sus bins de latitud y longitud, almacenándolos en metadatos para uso durante el entrenamiento

### 5.3 Generación de Embeddings (Opcional)

7. **Embeddings CLIP**: Pre-computamos embeddings CLIP para todas las imágenes del dataset, guardándolos en formato NPZ para acelerar el entrenamiento posterior

### 5.4 Entrenamiento y Evaluación

8. **Entrenamiento**: Entrenamos el modelo ViT con geolocalización y CLIP fusionado usando técnicas avanzadas de regularización
9. **Evaluación**: Generamos métricas completas incluyendo matriz de confusión y reporte de clasificación
10. **Predicción**: El modelo puede predecir el continente de nuevas imágenes, incluso sin coordenadas GPS

## 6. Técnicas Avanzadas de Regularización

Para combatir el overfitting y el desbalance de clases, implementamos varias técnicas:

### 6.1 Data Augmentation Agresiva

Usamos **Albumentations** con transformaciones más intensas:
- RandomResizedCrop con escala variable (0.7 a 1.0)
- ColorJitter aumentado (30% en brillo, contraste y saturación)
- Rotación hasta 20 grados
- Ruido gaussiano aleatorio para regularización adicional

### 6.2 Balanceo Adaptativo de Clases

Implementamos un sistema de **pesos adaptativos** que:
- **Detecta clases mayoritarias**: Si una clase tiene más del 30% del dataset, reduce su peso en 20%
- **Refuerza clases minoritarias**: Para África y Asia (si tienen menos del 15% del dataset), aumenta su peso en 30%
- **Usa WeightedRandomSampler**: Durante el entrenamiento, muestra más frecuentemente las clases minoritarias
- **Pérdida ponderada**: Además, la función de pérdida (CrossEntropyLoss) también está ponderada por los mismos pesos

### 6.3 Regularización contra Overfitting

- **Learning rate reducido**: 2e-5 en lugar de 3e-5 para entrenamiento más conservador
- **Weight decay aumentado**: 0.02 en lugar de 0.01 para mayor regularización L2
- **Dropout adicional**: 0.3 en el clasificador además del dropout estándar del ViT
- **Early stopping estricto**: Patience de 3 épocas en lugar de 5, con threshold más estricto
- **Detección automática**: El sistema compara métricas de train vs test al finalizar y alerta si detecta overfitting significativo (diferencia > 10%)

## 7. Dataset

Utilizamos el **Large Dataset of Geotagged Images** de Kaggle, que contiene más de 37 mil imágenes geoetiquetadas distribuidas en shards MessagePack. 

**Características del dataset**:
- **Distribución desbalanceada**: Europa y América dominan con aproximadamente el 84% de las imágenes, mientras que África y Oceanía están subrepresentadas
- **Calidad mejorada**: Filtramos solo imágenes outdoor con alta confianza, mejorando significativamente la calidad del dataset de entrenamiento
- **Integración incremental**: El sistema soporta agregar nuevos datos (como Google Street View) de manera incremental sin reprocesar todo

## 8. Resultados

### 8.1 Métricas Globales

**Con CLIP Fusionado**:
- **Accuracy**: 75.21%
- **F1-Score (weighted)**: 74.88%
- **Precision (weighted)**: 74.85%
- **Recall (weighted)**: 75.21%

**Sin CLIP (solo ViT + Geolocalización)**:
- **Accuracy**: 70.76%
- **F1-Score (weighted)**: 69.71%

La integración de CLIP proporciona una **mejora consistente de aproximadamente 5 puntos porcentuales** en todas las métricas.

### 8.2 Performance por Continente

El rendimiento varía según la disponibilidad de datos:

- **Europa**: Mejor rendimiento con F1-score de 0.83, debido a mayor cantidad de datos de entrenamiento
- **América**: F1-score de 0.78, con buen balance entre precision y recall
- **Asia**: F1-score de 0.66, con buena precisión pero recall moderado
- **Oceanía**: F1-score de 0.64, rendimiento estable
- **África**: F1-score de 0.58, la clase más desafiante debido a escasez de datos

### 8.3 Análisis de Resultados

El modelo muestra **buena capacidad de generalización** en clases mayoritarias y una capacidad razonable incluso en clases minoritarias. El desbalance del dataset se refleja en los resultados, pero las técnicas de balanceo implementadas ayudan a mitigar este problema.

## 9. Características Técnicas Destacadas

### 9.1 Pipeline Incremental

Todos los scripts de procesamiento soportan **modo incremental**, lo que significa que:
- Solo procesan datos nuevos
- Detectan automáticamente qué se ha procesado ya
- Permiten agregar nuevos datos sin reprocesar todo

Esto es especialmente útil cuando se agregan nuevos shards o datasets adicionales como Google Street View.

### 9.2 Integración de Google Street View

Desarrollamos un script especial (`integrate_streetview.py`) que permite integrar incrementalmente imágenes de Google Street View:
- **Limpieza robusta de coordenadas**: Maneja formatos europeos, valores extremos, y concatenaciones
- **Mapeo automático**: Mapea coordenadas a continentes automáticamente
- **Marca como outdoor**: Todas las imágenes de Street View se marcan automáticamente como outdoor
- **Evita duplicados**: Verifica nombres de archivo existentes para no reprocesar

### 9.3 Verificación de Data Leakage

Implementamos verificaciones exhaustivas para prevenir data leakage:
- **Verificación por nombre**: Asegura que no haya archivos duplicados por nombre entre train y test
- **Verificación por contenido**: Calcula hashes MD5 para detectar imágenes duplicadas incluso con nombres diferentes
- **Splits persistentes**: Los splits se generan una sola vez y se reutilizan, garantizando reproducibilidad

### 9.4 Detección Automática de Overfitting

El sistema automáticamente:
- Evalúa el modelo en el conjunto de entrenamiento después del entrenamiento
- Compara métricas de train vs test
- Emite advertencias si detecta diferencias significativas (>5% leve, >10% severo)
- Sugiere acciones correctivas (más regularización, más datos, menos épocas)

## 10. Contribuciones Técnicas

Este proyecto contribuye en varios aspectos:

1. **Integración multi-modal**: Demostramos cómo fusionar efectivamente características visuales (ViT), semánticas (CLIP), y geográficas (GPS bins) en un solo modelo
2. **Pipeline completo**: Proporcionamos un sistema end-to-end desde extracción de datos hasta predicción, con soporte incremental
3. **Balanceo adaptativo**: Implementamos un sistema inteligente de balanceo que se adapta automáticamente al desbalance del dataset
4. **Regularización avanzada**: Combinamos múltiples técnicas de regularización para combatir overfitting de manera efectiva

## 11. Desafíos Enfrentados y Soluciones

### 11.1 Desbalance de Clases

**Desafío**: Europa y América dominan el dataset, mientras que África y Oceanía tienen pocos ejemplos.

**Solución**: Implementamos balanceo adaptativo con pesos ajustados automáticamente y WeightedRandomSampler para muestrear más frecuentemente las clases minoritarias.

### 11.2 Overfitting

**Desafío**: El modelo mostró inicialmente una diferencia de más del 11% entre train y test.

**Solución**: Implementamos múltiples técnicas de regularización: learning rate reducido, weight decay aumentado, dropout adicional, early stopping más estricto, y data augmentation más agresiva.

### 11.3 Calidad del Dataset

**Desafío**: Las imágenes indoor tienen mucho ruido y reducen significativamente el rendimiento.

**Solución**: Utilizamos Places365 para filtrar solo imágenes outdoor con alta confianza, mejorando el F1-score de 52% a 75%.

### 11.4 Correspondencia CSV-Imágenes

**Desafío**: Durante la integración de Google Street View, mantener correspondencia 1:1 entre filas del CSV e imágenes después de filtrar coordenadas inválidas.

**Solución**: Implementamos un sistema de índices que preserva la correspondencia incluso después de filtrado múltiple.

## 12. Aplicaciones Prácticas

GeoVision puede aplicarse en múltiples contextos:

1. **Organización automática de fotos**: Clasificar colecciones fotográficas personales o profesionales por continente
2. **Verificación de contenido**: Verificar que las imágenes compartidas en redes sociales correspondan a su ubicación geográfica declarada
3. **Sistemas de recomendación**: Mejorar sistemas de recomendación basándose en preferencias geográficas
4. **Análisis de contenido turístico**: Analizar patrones de fotografías turísticas por continente
5. **Investigación geográfica**: Estudiar cómo diferentes regiones son representadas visualmente

## 13. Limitaciones y Trabajo Futuro

### 13.1 Limitaciones Actuales

- **Desbalance persistente**: Aunque mitigado, el desbalance de clases aún afecta el rendimiento en África y Oceanía
- **Solo continentes**: El modelo clasifica solo a nivel de continente, no de país o ciudad
- **Dependencia de outdoor**: Solo funciona bien con imágenes outdoor

### 13.2 Mejoras Futuras

- **Más datos**: Incorporar más shards o datasets adicionales para aumentar especialmente las clases minoritarias
- **Modelos más grandes**: Experimentar con ViT-Large o Swin Transformer para mejor rendimiento
- **Fine-tuning de CLIP**: En lugar de solo proyectar, hacer fine-tuning del modelo CLIP completo
- **Clasificación jerárquica**: Extender a país y ciudad, no solo continente
- **Explicabilidad**: Implementar Grad-CAM o attention maps para entender qué partes de la imagen son más importantes para la decisión
- **Features adicionales**: Incorporar OCR para texto en imágenes, hora del día, o información climática

## 14. Conclusiones

GeoVision demuestra que es posible clasificar imágenes geográficas con alta precisión utilizando técnicas modernas de deep learning. La integración exitosa de ViT, CLIP y features geográficas logra un **F1-score del 75%**, superando significativamente enfoques que usan solo características visuales.

Las técnicas de regularización y balanceo implementadas permiten manejar efectivamente el desbalance del dataset y prevenir overfitting, resultando en un modelo que generaliza bien a datos no vistos.

El pipeline completo desarrollado, con soporte incremental y múltiples verificaciones de calidad, proporciona una base sólida para aplicaciones prácticas y futuras mejoras.

---

## Notas para la Presentación

### Duración Estimada
- Presentación completa: 15-20 minutos
- Presentación resumida: 10 minutos (secciones 1-8 y 14)

### Puntos Clave para Énfasis
1. **Integración multi-modal**: Fusionar ViT + CLIP + Geolocalización
2. **Mejora con CLIP**: +5 puntos porcentuales en todas las métricas
3. **Balanceo adaptativo**: Sistema inteligente que se adapta al dataset
4. **Pipeline completo**: Sistema end-to-end con soporte incremental
5. **Resultados sólidos**: 75% F1-score en clasificación geográfica

### Preguntas Frecuentes Anticipadas
- **¿Por qué solo continentes?** Para mantener un nivel de abstracción que permita buena generalización. Clasificar países requeriría mucho más datos.
- **¿Funciona con cualquier imagen?** Funciona mejor con imágenes outdoor. Las imágenes indoor tienen mucho ruido geográfico.
- **¿Qué pasa si no tengo coordenadas GPS?** El modelo puede predecir sin coordenadas, aunque la precisión mejora cuando están disponibles.
- **¿Por qué CLIP ayuda tanto?** CLIP captura información semántica (objetos, escenas, contexto) que complementa las características visuales puras del ViT.

---

**Autores**: Agustín Figueroa y Esteban Alvarez  
**Repositorio**: https://github.com/Figs0203/GeoVision  
**Video del Proyecto**: https://youtu.be/8sCkD-1eMto?si=e50IRY_O9UGkO97z

