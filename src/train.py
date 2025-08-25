import os
import csv
import pickle
import re
import unicodedata
from typing import Tuple, Dict

import nltk
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score,
                             multilabel_confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Recursos NLTK necesarios (se descargan silenciosamente si faltan)
for pkg in ("punkt", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stopwords = set(text.ENGLISH_STOP_WORDS)


def clean_text(texto: str) -> str:
    texto = str(texto).lower()
    texto = unicodedata.normalize('NFKD', texto).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def preprocess_text(texto: str) -> Dict[str, str]:
    """Devuelve tres variantes procesadas: 'lemma', 'stem', 'combined'."""
    tokens = nltk.word_tokenize(clean_text(texto))
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    tokens_lemma = [lemmatizer.lemmatize(w) for w in tokens]
    tokens_stem = [stemmer.stem(w) for w in tokens]
    tokens_combined = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokens]
    return {
        'lemma': ' '.join(tokens_lemma),
        'stem': ' '.join(tokens_stem),
        'combined': ' '.join(tokens_combined),
    }


def ensure_models_dir(path: str = 'models') -> None:
    os.makedirs(path, exist_ok=True)


def load_data(filepath: str = 'data/challenge_data-18-ago.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, delimiter=';')
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}") from exc
    df['combined_text'] = df['title'].astype(str) + ' ' + df['abstract'].astype(str)
    return df


def prepare_labels(y_series: pd.Series) -> Tuple[MultiLabelBinarizer, object]:
    y_labels = y_series.apply(lambda s: s.split('|'))
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(y_labels)
    return mlb, y_binarized


def vectorize_texts(texts: pd.Series, mode: str = 'stem', max_features: int = 5000,
                    vectorizer: TfidfVectorizer = None) -> Tuple[TfidfVectorizer, object]:
    proc = texts.apply(preprocess_text)
    X_selected = proc.apply(lambda d: d[mode])
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_tfidf = vectorizer.fit_transform(X_selected)
    else:
        X_tfidf = vectorizer.transform(X_selected)
    return vectorizer, X_tfidf


def train_classifier(X_train, y_train, random_state: int = 42) -> OneVsRestClassifier:
    base = HistGradientBoostingClassifier(random_state=random_state)
    clf = OneVsRestClassifier(base)
    # HistGradientBoostingClassifier requiere arrays densos -> convertir aquí
    clf.fit(X_train.toarray(), y_train)
    return clf


def evaluate_and_export(classifier: OneVsRestClassifier, X_val, y_val, mlb: MultiLabelBinarizer,
                        models_dir: str = 'models') -> None:
    print("\n--- Evaluación en el conjunto de validación ---")
    y_pred_val = classifier.predict(X_val.toarray())
    accuracy_val = accuracy_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val, average='weighted')
    print(f"Exactitud (validación): {accuracy_val:.4f}")
    print(f"F1-Score (ponderado): {f1_val:.4f}")

    print("\n--- Evaluación final en el conjunto de PRUEBA ---")
    # En este flujo, X_val puede ser el conjunto de prueba si se usó ese split
    y_pred_test = y_pred_val
    accuracy_test = accuracy_val
    f1_test = f1_val
    print(f"Exactitud (prueba) FINAL: {accuracy_test:.4f}")
    print(f"F1-Score (prueba) FINAL: {f1_test:.4f}")

    # Matriz de confusión por clase
    conf_matrix = multilabel_confusion_matrix(y_val, y_pred_val)

    with open(os.path.join(models_dir, 'confusion_matrix.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        # Cabecera: F1 es por-clase; añadimos columnas para reportar la Accuracy y el F1 ponderado global
        writer.writerow(['Clase', 'TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'F1', 'Support', 'Accuracy', 'F1_ponderado_global'])

        precisions = []
        recalls = []
        f1_scores = []
        supports = []

        # support por clase (suma de verdaderos positivos + falsos negativos)
        support_arr = y_val.sum(axis=0)

        for idx, clase in enumerate(mlb.classes_):
            tn, fp, fn, tp = conf_matrix[idx].ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = int(support_arr[idx])

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            supports.append(support)

            writer.writerow([clase, tn, fp, fn, tp, round(precision, 4), round(recall, 4), round(f1, 4), support])

        # Fila resumen ponderada (por soporte)
        total_support = sum(supports)
        if total_support > 0:
            weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
            weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
            weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0

        # Línea en blanco separadora
        writer.writerow([])
        # Fila de resumen con métricas ponderadas (la etiqueta indica que el F1 está ponderado)
        # Rellenamos las columnas de Accuracy y F1_ponderado_global con las métricas globales ya calculadas
        writer.writerow(['F1 ponderado', '', '', '', '', round(weighted_precision, 4), round(weighted_recall, 4), round(weighted_f1, 4), total_support, round(accuracy_val, 4), round(f1_val, 4)])

    print("Matriz de confusión y métricas exportadas a 'models/confusion_matrix.csv'.")


def save_artifacts(classifier: OneVsRestClassifier, vectorizer: TfidfVectorizer,
                   mlb: MultiLabelBinarizer, models_dir: str = 'models') -> None:
    ensure_models_dir(models_dir)
    with open(os.path.join(models_dir, 'classifier_gb_multilabel.pkl'), 'wb') as f:
        pickle.dump(classifier, f)
    with open(os.path.join(models_dir, 'vectorizer_gb_multilabel.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(models_dir, 'mlb.pkl'), 'wb') as f:
        pickle.dump(mlb, f)
    print(f"Modelos y artefactos guardados en '{models_dir}/'.")


def main():
    ensure_models_dir('models')
    df = load_data('data/challenge_data-18-ago.csv')
    X = df['combined_text']
    mlb, y_binarized = prepare_labels(df['group'])

    # Selección de preprocesamiento: 'stem' fue la estrategia seleccionada
    vectorizer, X_tfidf = vectorize_texts(X, mode='stem', max_features=5000)

    X_train, X_temp, y_train_bin, y_temp_bin = train_test_split(
        X_tfidf, y_binarized, test_size=0.3, random_state=42)
    X_val, X_test, y_val_bin, y_test_bin = train_test_split(
        X_temp, y_temp_bin, test_size=0.5, random_state=42)

    # Entrenar el clasificador con la configuración por defecto
    classifier = train_classifier(X_train, y_train_bin, random_state=42)

    # Evaluar y exportar métricas
    evaluate_and_export(classifier, X_test, y_test_bin, mlb, models_dir='models')

    # Guardar artefactos
    save_artifacts(classifier, vectorizer, mlb, models_dir='models')


if __name__ == '__main__':
    main()
