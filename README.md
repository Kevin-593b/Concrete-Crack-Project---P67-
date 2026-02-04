# Proyecto Final – Visión por Computador (Grupo 6)
##  Integrantes (Grupo 6)
- Robinson Agual  
- Bryan Barrera  
- Kevin Bonilla  
- Ricardo Maigua  
- Xavier Rosillo  
### Notas importantes
- El proyecto fue desarrollado y ejecutado en Google Colab (CPU), por lo que el flujo se ejecuta por celdas.
- Para reproducir resultados, se debe mantener la estructura del dataset:
  - `/dataset/final_dataset_concreto/train/{Negative,Positive}`
  - `/dataset/final_dataset_concreto/test/{Negative,Positive}`
- Si cambias la ubicación en Drive, actualiza las rutas en las primeras celdas de cada sección (`PROJECT_ROOT`, `DATASET_DIR`).

---
**Nombre del Proyecto:** Diseño, implementación y evaluación de un sistema de clasificación de imágenes mediante visión por computador clásica y aprendizaje profundo.

## 1. Descripción general del proyecto
Este proyecto consiste en diseñar, implementar y evaluar un sistema de **clasificación binaria** de imágenes de superficies de concreto, comparando el desempeño entre:
- **Visión por computador clásica** (extracción manual de características + clasificadores tradicionales).
- **Aprendizaje profundo** (CNN entrenada con imágenes originales).

El objetivo es distinguir entre:
- **Negative:** ausencia de grietas o fisuras mínimas.
- **Positive:** presencia de grietas o fisuras evidentes.

---

## 2. Dataset
Dataset público obtenido de Kaggle:

- **Nombre:** *Concrete Crack Images for Classification*  
- **Enlace:** https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification  
- **Descripción:** imágenes de superficies de concreto capturadas en edificios del campus de la **Middle East Technical University (METU)**.  
- **Clases:** 2 (Positive / Negative)

### Subconjunto utilizado (balanceado)
Para ejecución eficiente en Google Colab, se trabajó con un subconjunto filtrado y balanceado:

- **Total global:** 5,000 imágenes  
- **Train:** 4,000 (2,000 Negative + 2,000 Positive)  
- **Test:** 1,000 (500 Negative + 500 Positive)

Para la CNN, el conjunto **train** se subdividió internamente:
- **Training interno:** 3,200 (80% del train)
- **Validation:** 800 (20% del train)
- **Test:** 1,000 (evaluación final)

---

## 3. Estructura del repositorio
Estructura recomendada y utilizada:

- `/dataset/final_dataset_concreto/`
  - `/train/Negative`
  - `/train/Positive`
  - `/test/Negative`
  - `/test/Positive`

- `/src/`
  - `/features`  → archivos `.npz` con características extraídas (HOG, Hu, Zernike, GLCM)
  - `/models`    → modelos entrenados clásicos (`.joblib`) y CNN (`.keras`)
  - `/cnn`       → historial de entrenamiento CNN (`.json`)

- `/results/`    → métricas, matrices de confusión, comparaciones finales, etc.

- `README.md`

---

## 4. Metodología (resumen)
### 4.1 Preprocesamiento y análisis exploratorio
- Conteo y verificación de balance por clase.
- Visualización de muestras por clase.
- Histogramas de intensidades/colores.
- Mejora de contraste (ej. CLAHE) para análisis exploratorio.

### 4.2 Extracción de características (mínimo 3)
Se implementaron 4 métodos:
- **HOG (Histogram of Oriented Gradients)**
- **Momentos de Hu**
- **Momentos de Zernike**
- **GLCM/Haralick (textura)**

Las características se guardan como `.npz` dentro de `/src/features/`:
- `hog_features.npz`
- `hu_features.npz`
- `zernike_features_deg6.npz`
- `glcm_haralick_features.npz`

Cada archivo incluye: `X_train`, `y_train`, `X_test`, `y_test`.

### 4.3 Clasificadores tradicionales (2 modelos)
Se entrenaron y evaluaron:
- **SVM lineal (LinearSVC)** con estandarización (`StandardScaler`).
- **Random Forest**.

Se guardan tablas de resultados (CSV) y matrices de confusión (PNG) en `/results/`.

### 4.4 CNN (aprendizaje profundo)
Se implementó una CNN mediante **transfer learning** con **EfficientNetB0**:
- Imágenes reescaladas a 224×224.
- Data augmentation (flip, rotación, zoom, contraste).
- Pérdida: `binary_crossentropy`
- Optimizador: `Adam`
- Métricas: accuracy, AUC, precision, recall
- Entrenamiento en dos fases: *freeze* + *fine-tuning*.

Se guardan: modelo `.keras`, history `.json`, métricas `.json`, y predicciones `.npy`.

### 4.5 Evaluación y comparación (3.6)
Se consolidan resultados de:
- Modelos clásicos (por extractor de características)
- CNN (imágenes crudas)

Se genera una **tabla comparativa final** (CSV) y se discuten trade-offs entre desempeño y costo computacional.

---

## 5. Librerías utilizadas
Recomendado ejecutar en **Google Colab**.

Principales dependencias:
- `numpy`, `pandas`
- `matplotlib`
- `opencv-python`
- `scikit-image`
- `scikit-learn`
- `tensorflow` / `keras`
- `mahotas` (Zernike)
- `joblib`


Instalación sugerida (Colab): `pip install -U numpy pandas matplotlib opencv-python scikit-image scikit-learn joblib && 
pip install mahotas==1.4.12`



## 6. Ejecución (Google Colab por secciones/celdas)
Este proyecto fue desarrollado ejecutando el flujo **por partes en Google Colab**, por lo que no existe un único script de ejecución.

Flujo recomendado:
1. Montar Google Drive y definir rutas del proyecto.
2. Verificar y analizar el dataset (conteos, muestras, histogramas).
3. Extraer características y guardar archivos `.npz` en `/src/features`:
   - HOG, Hu, Zernike, GLCM/Haralick.
4. Entrenar clasificadores tradicionales (SVM lineal y Random Forest) usando los `.npz` y guardar:
   - métricas en `.csv` y matrices de confusión en `.png` dentro de `/results`.
5. Entrenar la CNN (EfficientNetB0 Transfer Learning) usando imágenes originales:
   - entrenamiento + validación (desde `train`), evaluación final en `test`,
   - guardar modelo `.keras`, history `.json`, métricas y predicciones en `/results`.
6. Generar la tabla comparativa final (3.6) integrando:
   - resultados clásicos (por extractor) + resultados CNN.

---

## 7. Resultados generados (archivos clave)
- **Features extraídas:** `/src/features/*.npz`
- **Modelos clásicos entrenados:** `/src/models/*.joblib`
- **Modelo CNN entrenado:** `/src/models/*.keras`
- **Métricas clásicos:** `/results/resultados_3_4_*.csv`
- **Métricas CNN:** `/results/resultados_3_5_cnn_*.json`
- **Comparación final:** `/results/resultados_3_6_*.csv`
- **Matrices de confusión:** `/results/cm_*.png`
- **Predicciones CNN (si aplica):** `/results/cnn_y_true_*.npy`, `/results/cnn_y_pred_*.npy`, `/results/cnn_y_prob_*.npy`

---
## Nota sobre archivos omitidos (tamaño del repositorio)

Por limitaciones de tamaño en GitHub, **no se incluyó el dataset completo** (`/dataset`) ni algunos archivos generados de gran peso (por ejemplo, ciertas **características extraídas** en formato `.npz`, predicciones `.npy`, y/o algunos resultados intermedios). 

El **dataset original** puede descargarse desde Kaggle (ver enlace en la sección *Dataset*), y los **resultados completos (tablas, matrices de confusión, curvas y métricas)** se encuentran documentados en el **informe final** del proyecto.


> **Nota:** En este repositorio también se incluyen los archivos **Proyecto_Vision_Concreto_v3.ipynb** (notebook principal de Google Colab/Jupyter) y **Proyecto_Vision_Concreto_v3.html** (exportación del notebook), donde se encuentra el flujo completo ejecutado por celdas: análisis exploratorio, extracción de características, entrenamiento de clasificadores tradicionales, entrenamiento de la CNN y la comparación final de resultados.

