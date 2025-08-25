import pandas as pd
import pickle
import sys
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from src.preprocessing import preprocess

def format_labels(labels):
    """Convierte lista de etiquetas en string, o indica si no hay grupo."""
    return "|".join(labels) if labels else "no group found"


def main():
    """Carga un CSV, predice grupos y muestra métricas. Permite omitir la evaluación si no hay etiquetas verdaderas.

    Uso:
      python app.py --input data/nuevo_dataset.csv --output models/predicciones_evaluadas.csv --delimiter ';' [--skip-eval]
    """
    import argparse

    parser = argparse.ArgumentParser(description='Evaluar predicción de grupos en un CSV')
    parser.add_argument('--input', type=str, required=True, help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', type=str, default='models/predicciones_evaluadas.csv', help='Ruta al archivo CSV de salida')
    parser.add_argument('--delimiter', type=str, default=';', help="Delimitador del CSV de entrada (por defecto ';')")
    parser.add_argument('--skip-eval', action='store_true', help='No calcular métricas aunque exista la columna group')
    args = parser.parse_args()

    df = pd.read_csv(args.input, delimiter=args.delimiter)
    # Requerir al menos title y abstract
    required_cols = ['title', 'abstract']
    if not all(col in df.columns for col in required_cols):
        print(f'El archivo debe tener las columnas: {required_cols}')
        sys.exit(1)

    # Preprocesamiento modular
    X_new = (df['title'].astype(str) + ' ' + df['abstract'].astype(str)).apply(preprocess)

    # Cargar vectorizador, modelo y binarizador
    with open('models/vectorizer_gb_multilabel.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/classifier_gb_multilabel.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)

    # Transformar y predecir
    X_new_tfidf = vectorizer.transform(X_new)
    y_pred = model.predict(X_new_tfidf.toarray())
    labels_pred = mlb.inverse_transform(y_pred)
    df['group_predicted'] = [format_labels(labels) for labels in labels_pred]

    # Evaluación (opcional) si existe columna 'group' y no se solicitó omitir
    if 'group_predicted' in df.columns and not args.skip_eval:
        true_lists = df['group_predicted'].fillna('').apply(lambda s: s.split('|') if s else [])
        # Filtrar etiquetas desconocidas para evitar errores con mlb.transform
        filtered_true = true_lists.apply(lambda labels: [l for l in labels if l in mlb.classes_])
        if any(len(orig) != len(filt) for orig, filt in zip(true_lists, filtered_true)):
            print('Aviso: algunas etiquetas verdaderas no estaban en el conjunto de entrenamiento y se ignoraron para la evaluación.')

        y_true_bin = mlb.transform(filtered_true.tolist())

        # Métricas principales (AHORA calculadas con predicted_group como referencia)
        # Intercambiamos roles: tratamos las predicciones como referencia para obtener métricas "por grupo predicho"
        acc_pred = accuracy_score(y_pred, y_true_bin)
        f1w_pred = f1_score(y_pred, y_true_bin, average='weighted', zero_division=0)
        print(f'Exactitud (subset accuracy) usando predicted_group como referencia: {acc_pred:.4f}')
        print(f'F1-score ponderado (global) usando predicted_group como referencia: {f1w_pred:.4f}')

        print('\nReporte por grupo predicho:')
        # clasificación con roles invertidos: primero las predicciones como 'y_true'
        print(classification_report(y_pred, y_true_bin, target_names=mlb.classes_, zero_division=0))

        print('\nMatriz de confusión multilabel (predicted vs true):')
        conf_matrix_pred = multilabel_confusion_matrix(y_pred, y_true_bin)
        for idx, clase in enumerate(mlb.classes_):
            print(f'Clase (predicha): {clase}')
            print(conf_matrix_pred[idx])
    else:
        print('Evaluación omitida: no hay columna "group" o se indicó --skip-eval. Solo se generaron predicciones.')

    # Exportar resultados con la columna group_predicted
    df.to_csv(args.output, index=False)
    print(f'Resultados exportados a {args.output}')


if __name__ == '__main__':
    main()
