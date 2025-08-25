# Clasificación de Literatura Médica con Machine Learning

Este proyecto implementa una solución de **Machine Learning** para la clasificación automática de literatura médica.  
Se utiliza procesamiento de lenguaje natural y modelos de clasificación multietiqueta.

**Informe y Diagrama**  
- ![Ver diagrama](ruta/a/la/imagen.png)
- [Ver informe final](ruta/a/archivo.pdf)

## Requerimientos

* Python 3.8+
* pandas
* scikit-learn
* nltk

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Entrenamiento
```bash
python src/train.py
```

## Ejecución

El script `app.py` predice el/los grupo(s) para cada fila del CSV (columnas `title`, `abstract`)

Requisitos previos
- Ejecutar desde la raíz del proyecto.
- Tener instaladas las dependencias del proyecto (p. ej. `pandas`, `scikit-learn`, `nltk`).
- Tener los artefactos en `models/`:
  - `vectorizer_gb_multilabel.pkl`
  - `classifier_gb_multilabel.pkl`
  - `mlb.pkl`

Formato de entrada

- Predecir y evaluar (CSV con `;` y columna `group_predicted` presente):
 ```bash
  python app.py --input data/nuevo_dataset.csv --output output/predicciones_evaluadas.csv --delimiter ';'
  ```
  
- Para guardar también las métricas impresas en un archivo plano:
 ```bash
  python app.py --input data/nuevo_dataset.csv --output models/predicciones_evaluadas.csv --delimiter ';' > output/predicciones_evaluadas.txt
  ```
  
- CSV con comas como separador:
```bash
python app.py --input data/nuevo_dataset_comma.csv --delimiter , --output output/predicciones_evaluadas.csv
```
  

Produce un archivo de salida csv las columnas originales + `group_predicted` (ruta = `--output`).


Notas
- Si algunas etiquetas verdaderas no fueron vistas en entrenamiento, se ignoran durante la evaluación.

---


---

##  Flujo de la Solución

### 1. Carga de datos
- **Lectura CSV:**  
  ```python
  pd.read_csv()
```

* **Combinación de columnas:**

  ```python
  df['title'].astype(str) + " " + df['abstract'].astype(str)
  ```

---

### 2. Preprocesamiento de texto

* **Conversión a minúsculas:**

  ```python
  str(texto).lower()
  ```
* **Eliminación de caracteres especiales:**

  ```python
  unicodedata.normalize(), re.sub()
  ```
* **Tokenización:**

  ```python
  nltk.word_tokenize()
  ```
* **Eliminación de stopwords:**

  ```python
  word not in stopwords
  ```
* **Lematización:**

  ```python
  WordNetLemmatizer().lemmatize()
  ```
* **Stemming:**

  ```python
  SnowballStemmer().stem()
  ```

---

### 3. Vectorización

* **TF-IDF:**

  ```python
  TfidfVectorizer(max_features=5000)
  ```
* **Transformación:**

  ```python
  vectorizer.fit_transform()
  ```

---

### 4. Binarización de etiquetas

* **MultiLabelBinarizer:**

  ```python
  MultiLabelBinarizer(), mlb.fit_transform()
  ```

---

### 5. División de datos

* **train\_test\_split:**

  ```python
  train_test_split()
  ```

---

### 6. Entrenamiento del modelo

* **Definición del modelo:**

  ```python
  HistGradientBoostingClassifier()
  ```
* **Clasificador multietiqueta:**

  ```python
  OneVsRestClassifier()
  ```
* **Ajuste:**

  ```python
  clasificador_multietiqueta.fit()
  ```

---

### 7. Evaluación

* **Predicción:**

  ```python
  clasificador_multietiqueta.predict()
  ```
* **Accuracy:**

  ```python
  accuracy_score()
  ```
* **F1-Score:**

  ```python
  f1_score()
  ```

---

### 8. Análisis por clase y exportación

* **Matriz de confusión:**

  ```python
  multilabel_confusion_matrix()
  ```
* **Exportar a CSV:**

  ```python
  csv.writer(), writer.writerow()
  ```

---


## Resultados

El modelo entrega métricas de **accuracy** y **F1-Score ponderado** en los conjuntos de validación y prueba, junto con un análisis por grupo.
