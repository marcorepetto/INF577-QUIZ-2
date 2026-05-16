# Propuestas de Mejora para el Modelo de Rostros

Basado en la literatura de procesamiento de imágenes y reconocimiento facial, se proponen las siguientes mejoras para cerrar la brecha entre la validación local y el score de Kaggle (0.69209).

## 1. Arquitectura del Modelo (Backbone)
La CNN actual es muy básica para capturar la complejidad de los atributos faciales.
- **Transfer Learning**: Utilizar modelos pre-entrenados en ImageNet como **ResNet-18/34/50** o **EfficientNet-B0**. Estos modelos ya tienen filtros especializados en bordes, texturas y formas complejas.
- **Arquitecturas Especializadas**: Investigar **MobileFaceNet** o **ShuffleFaceNet**, que están optimizadas específicamente para procesar rostros con pocos parámetros pero alta precisión.
- **Attention Modules**: Implementar bloques **Squeeze-and-Excitation (SE)** para permitir que la red se enfoque en los canales (características) más importantes de la imagen.

## 2. Aumento de Datos y Regularización
El modelo actual parece estar sobreajustando. Un aumento de datos más agresivo obligará al modelo a aprender características más robustas.
- **Augmentations Avanzadas**:
    - [x] **ColorJitter**: Variar brillo, contraste, saturación y tono (implementado).
    - [ ] **RandomRotation**: Rotaciones pequeñas (+/- 15 grados).
    - [x] **GaussianBlur**: Simular fotos desenfocadas (implementado).
    - [ ] **RandomResizedCrop**: Ayuda a que el modelo sea invariante a la posición exacta del rostro.
- **Mixup / CutMix**: Técnicas que combinan dos imágenes y sus etiquetas durante el entrenamiento, mejorando significativamente la generalización.
- [x] **Label Smoothing**: En lugar de etiquetas duras (0 o 1), usar valores suavizados para reducir la confianza excesiva del modelo (implementado).

## 3. Preprocesamiento de Imágenes
- **Normalización**: Aplicar normalización `(x - mean) / std`. Si se usa transfer learning, usar los valores de ImageNet. Si no, calcularlos sobre el dataset de entrenamiento.
- **Alineación Facial**: Si el dataset lo permite, detectar puntos de referencia (landmarks) y alinear los rostros. Esto reduce la varianza espacial y facilita mucho la tarea de clasificación.

## 4. Estrategia de Entrenamiento
- **Scheduler**: Cambiar a **OneCycleLR** o **CosineAnnealingLR**. Estos facilitan encontrar mínimos más profundos y estables comparado con `ReduceLROnPlateau`.
- **Warmup**: Empezar con una LR muy baja por unas pocas épocas para evitar desestabilizar los pesos al inicio (especialmente con transfer learning).
- **Focal Loss**: Si existen ejemplos de rostros muy difíciles de clasificar, Focal Loss asigna más peso a esos errores que a los ejemplos fáciles.

## 5. Validación y Post-procesamiento
- **K-Fold Cross Validation**: En lugar de un solo split 80/20, usar 5-folds para asegurar que el modelo sea robusto a diferentes particiones de los datos.
- **Test-Time Augmentation (TTA)**: Durante la inferencia, pasar la misma imagen y su versión "flipped", promediando los resultados. Esto suele subir el score en Kaggle entre 1% y 3%.
- **Ensemble**: Entrenar dos arquitecturas diferentes (ej. ResNet y EfficientNet) y promediar sus predicciones.

## 6. Diagnóstico de la Brecha de Score
La diferencia entre 0.83 (val) y 0.69 (Kaggle) es muy alta.
- **Verificar el Balance**: Asegurarse de que el umbral (threshold) de 0.5 es óptimo. A veces, ajustar el umbral en el set de validación mejora el F1.
- **Consistencia de Preprocesamiento**: Garantizar que el preprocesamiento aplicado en el set de prueba sea **idéntico** al de validación (especialmente el escalado a [0, 1]).
