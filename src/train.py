import csv
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import unicodedata
import nltk
from sklearn.feature_extraction import text
from nltk.stem import SnowballStemmer, WordNetLemmatizer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def clean_text(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize('NFKD', texto).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


stopwords = set(text.ENGLISH_STOP_WORDS)


def preprocess_text(texto):
    # Tokeniza
    tokens = nltk.word_tokenize(clean_text(texto))
    # Solo lematización
    tokens_lemma = [lemmatizer.lemmatize(
        word) for word in tokens if word not in stopwords and len(word) > 2]
    # Solo stemming
    tokens_stem = [stemmer.stem(
        word) for word in tokens if word not in stopwords and len(word) > 2]
    # Combinado
    tokens_combined = [stemmer.stem(lemmatizer.lemmatize(
        word)) for word in tokens if word not in stopwords and len(word) > 2]
    return {
        'lemma': ' '.join(tokens_lemma),
        'stem': ' '.join(tokens_stem),
        'combined': ' '.join(tokens_combined)
    }


# --- 1. Cargar y preparar los datos ---
try:
    df = pd.read_csv('data/challenge_data-18-ago.csv', delimiter=';')
except FileNotFoundError:
    print("Error: El archivo 'challenge_data-18-ago.csv' no fue encontrado.")
    exit()

df['combined_text'] = df['title'].astype(
    str) + " " + df['abstract'].astype(str)
X = df['combined_text']
y = df['group']
y_labels = y.apply(lambda s: s.split('|'))

mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform(y_labels)

print("Etiquetas originales y su representación binaria:")
print(f"Original: {y_labels.iloc[0]} -> Binario: {y_binarized[0]}")
print(f"Clases únicas: {mlb.classes_}")

# --- Preprocesamiento con lematización y stemming combinados ---

# Preprocesa los textos en los tres modos

# Preprocesamiento solo con stemming
X_proc_dict = X.apply(preprocess_text)
X_stem = X_proc_dict.apply(lambda d: d['stem'])

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X_stem)
X_train, X_temp, y_train_bin, y_temp_bin = train_test_split(
    X_tfidf, y_binarized, test_size=0.3, random_state=42)
X_val, X_test, y_val_bin, y_test_bin = train_test_split(
    X_temp, y_temp_bin, test_size=0.5, random_state=42)
modelo_gb = HistGradientBoostingClassifier(random_state=42)
clasificador_multietiqueta = OneVsRestClassifier(modelo_gb)
clasificador_multietiqueta.fit(X_train.toarray(), y_train_bin)

# --- 4. Entrenar el clasificador con Gradient Boosting ---
print("\n--- Entrenando el clasificador con TF-IDF + Gradient Boosting ---")
modelo_gb = HistGradientBoostingClassifier(random_state=42)
clasificador_multietiqueta = OneVsRestClassifier(modelo_gb)

clasificador_multietiqueta.fit(X_train.toarray(), y_train_bin)

# --- 5. Evaluar el modelo ---
print("\n--- Evaluación en el conjunto de validación ---")
y_pred_val = clasificador_multietiqueta.predict(X_val.toarray())
accuracy_val = accuracy_score(y_val_bin, y_pred_val)
f1_val = f1_score(y_val_bin, y_pred_val, average='weighted')

print(f"Exactitud (Subconjunto): {accuracy_val:.2f}")
print(f"F1-Score (promedio ponderado): {f1_val:.2f}")

print("\n--- Evaluación final en el conjunto de PRUEBA ---")
y_pred_test = clasificador_multietiqueta.predict(X_test.toarray())
accuracy_test = accuracy_score(y_test_bin, y_pred_test)
f1_test = f1_score(y_test_bin, y_pred_test, average='weighted')

print(f"Exactitud (Subconjunto) FINAL: {accuracy_test:.2f}")
print(f"F1-Score (promedio ponderado) FINAL: {f1_test:.2f}")

# --- Guardar el modelo y el vectorizador ---
with open('models/classifier_gb_multilabel.pkl', 'wb') as f:
    pickle.dump(clasificador_multietiqueta, f)
with open('models/vectorizer_gb_multilabel.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Modelo GradientBoosting multietiqueta y vectorizador guardados en la carpeta 'models/'.")
print(
    f"Precisión final del modelo GradientBoosting multietiqueta: {accuracy_test:.2f}")

# --- Matriz de confusión multietiqueta ---
print("\n--- Matriz de confusión por clase (conjunto de prueba) ---")
conf_matrix = multilabel_confusion_matrix(y_test_bin, y_pred_test)
for idx, clase in enumerate(mlb.classes_):

    print(f"\nClase: {clase}")
    print(conf_matrix[idx])
    tn, fp, fn, tp = conf_matrix[idx].ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

# --- Exportar métricas y matriz de confusión a archivos CSV ---
with open('models/confusion_matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Clase', 'TN', 'FP', 'FN', 'TP',
                    'Precision', 'Recall', 'F1-Score'])
    for idx, clase in enumerate(mlb.classes_):
        tn, fp, fn, tp = conf_matrix[idx].ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0
        writer.writerow([clase, tn, fp, fn, tp, round(
            precision, 4), round(recall, 4), round(f1, 4)])
print("Matriz de confusión y métricas exportadas a 'models/confusion_matrix.csv'.")
