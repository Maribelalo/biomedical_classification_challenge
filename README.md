# Clasificación de Literatura Médica con Machine Learning

Este proyecto implementa una solución de **Inteligencia Artificial** para la clasificación automática de literatura médica.  
Se utiliza procesamiento de lenguaje natural (NLP) y modelos de clasificación multietiqueta.

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

### 5. ✂División de datos

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

## Ejecución

```bash
python main.py
```

---

## Resultados

El modelo entrega métricas de **accuracy** y **F1-Score ponderado** en los conjuntos de validación y prueba, junto con un análisis por clase.
