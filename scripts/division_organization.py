import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Diretório de origem dos arquivos
input_dir = '/home/joaovfg/PFC-WPW/mit-bih-segmented-signals/'
# Diretório de saída
output_base = '/home/joaovfg/PFC-WPW/mit-bih-organized/'

# Proporções
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Garante que o diretório de saída esteja limpo
if os.path.exists(output_base):
    shutil.rmtree(output_base)
os.makedirs(output_base)

# Coleta os arquivos pares .dat e .npy
pairs = []
for file in os.listdir(input_dir):
    if file.startswith('segments_') and file.endswith('.dat'):
        number = file.split('_')[1].split('.')[0]
        label_file = f'labels_{number}.npy'
        if label_file in os.listdir(input_dir):
            pairs.append((os.path.join(input_dir, file), os.path.join(input_dir, label_file)))

# Função auxiliar para salvar arquivos segmentados
def save_segment(output_dir, base_name, index, segment, label):
    class_dir = os.path.join(output_dir, f'class_{label}')
    os.makedirs(class_dir, exist_ok=True)
    segment_file = os.path.join(class_dir, f'{base_name}_seg{index}.dat')
    segment.astype(np.float32).tofile(segment_file)

# Processa os arquivos
for seg_path, label_path in pairs:
    base_name = os.path.basename(seg_path).split('.')[0]  # ex: segments_1
    labels = np.load(label_path)
    segments = np.fromfile(seg_path, dtype=np.float32)
    # Suponha que cada segmento tem tamanho fixo
    n_segments = len(labels)
    segment_size = len(segments) // n_segments
    segments = segments.reshape((n_segments, segment_size))

    # Divide entre treino, validação e teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        segments, labels, test_size=test_ratio, stratify=labels, random_state=42)
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42)

    # Função para salvar um conjunto
    def save_set(X, y, set_name):
        for i, (segment, label) in enumerate(zip(X, y)):
            save_segment(os.path.join(output_base, set_name), base_name, i, segment, label)

    save_set(X_train, y_train, 'train')
    save_set(X_val, y_val, 'validation')
    save_set(X_test, y_test, 'test')
