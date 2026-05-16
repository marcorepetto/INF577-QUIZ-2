# Documentación del Modelo Actual

Este documento detalla las características técnicas del modelo utilizado para el Quiz 2.

## 1. Arquitectura

### Backbone: CNN Personalizada
El backbone es una red convolucional simple definida en `models/backbone/CNN.py`:
- **Stem**: Convolución 3x3 (32 filtros, stride 1, padding 1) -> Batch Normalization -> ReLU.
- **Bloque 1**: 
    - Conv 3x3 (32 filtros) -> BN -> ReLU.
    - Conv 3x3 (32 filtros) -> BN -> ReLU.
    - MaxPool 2x2.
- **Bloque 2**:
    - Conv 3x3 (64 filtros) -> BN -> ReLU.
    - Conv 3x3 (64 filtros) -> BN -> ReLU.
    - MaxPool 2x2.
- **Bloque 3**:
    - Conv 3x3 (128 filtros) -> BN -> ReLU.
    - Conv 3x3 (128 filtros) -> BN -> ReLU.
- **Global Average Pooling**: Reduce los mapas de características a un vector de tamaño 128.

### Head: MLP (Multi-Layer Perceptron)
El clasificador final definido en `models/heads/MLP.py`:
- **Capa Lineal**: 128 neuronas de entrada -> 64 neuronas de salida.
- **Dropout**: Probabilidad de 0.5.
- **Activación**: ReLU.
- **Capa de Salida**: 64 neuronas -> 1 neurona (salida lineal para `BCEWithLogitsLoss`).

## 2. Entrenamiento y Optimización

- **Optimizador**: Adam.
    - Tasa de aprendizaje (LR): 0.001.
    - Betas: [0.9, 0.999].
    - Weight Decay: 0.01 (L2 Regularization).
- **Scheduler**: `ReduceLROnPlateau`.
    - Factor: 0.5.
    - Paciencia: 5 épocas.
    - Métrica monitoreada: Val F1 (maximizando).
- **Early Stopping**: 
    - Paciencia: 10 épocas.
    - Min Delta: 0.0001.
    - Detiene el entrenamiento si el F1 de validación no mejora significativamente, evitando el sobreajuste.
- **Loss Function**: `BCEWithLogitsLoss`.
    - Incluye `pos_weight` calculado dinámicamente según la proporción de clases en el set de entrenamiento para mitigar el desbalance.
- **Label Smoothing**: Aplicado un factor de 0.1 (transforma 0 -> 0.05 y 1 -> 0.95 para suavizar las etiquetas y evitar sobreconfianza).

## 3. Datos y Preprocesamiento

- **Balance de Clases**: 
    - División estratificada (80% train, 20% val).
    - Pesos en la función de pérdida (`pos_weight`).
- **Aumento de Datos (Augmentation)**:
    - `RandomHorizontalFlip`: Activado por defecto.
    - `ColorJitter`: Brillo (0.2), Contraste (0.2), Saturación (0.2), Hue (0.1).
    - `GaussianBlur`: Kernel size 3, Sigma [0.1, 2.0].
    - Configurable dinámicamente mediante Hydra en `config/config.yaml`.
- **Normalización**: 
    - Las imágenes se escalan de `[0, 255]` a `[0, 1]` dividiendo por 255. No se aplica normalización por media/desviación estándar.
- **Dimensiones de Entrada**: 64x64 píxeles, 3 canales (RGB).

## 4. Rendimiento Observado
- **Mejor F1 en Validación**: 0.8366.
- **Puntuación en Kaggle**: 0.69209.
- **Observación**: Existe una brecha significativa (~14%) entre la validación local y el score de Kaggle, lo que sugiere un posible sobreajuste (overfitting) o una diferencia en la distribución de los datos de prueba.
