import pandas as pd
from collections import Counter

df = pd.read_csv('data/challenge_data-18-ago.csv', delimiter=';')

# Extraer todas las etiquetas únicas
groups = df['group'].dropna().apply(lambda s: s.split('|'))
all_labels = set()
for labels in groups:
    all_labels.update(labels)
print('Etiquetas únicas:', all_labels)

# Contar frecuencia de cada etiqueta
label_counts = Counter()
for labels in groups:
    for label in labels:
        label_counts[label] += 1
print('Frecuencia por grupo:')
for label, count in label_counts.items():
    print(f'{label}: {count}')

# Proporción de artículos multietiqueta
multi_label_count = groups.apply(lambda x: len(x) > 1).sum()
print(f'Artículos multietiqueta: {multi_label_count} de {len(df)} ({multi_label_count/len(df):.2%})')

# Exportar resumen a CSV
summary = pd.DataFrame({'Grupo': list(label_counts.keys()), 'Frecuencia': list(label_counts.values())})
summary.to_csv('models/group_distribution.csv', index=False)
print("Resumen de distribución de grupos exportado a 'models/group_distribution.csv'.")
