import pandas as pd
import numpy as np
import wfdb
import os
from tqdm import tqdm

# Configurações do segmento
window_left = 76
window_right = 140
segment_length = window_left + window_right

# Caminho onde estão os registros MIT-BIH
local_path = '/home/joaovfg/PFC-WPW/mit-bih-arrhythmia-database-1.0.0/'  # ajuste se necessário

# Lê o DataFrame com os batimentos de interesse
df = pd.read_csv('/home/joaovfg/PFC-WPW/dataFrames/batimentos_filtrados.csv')

# Agrupa batimentos por registro
registros_unicos = df['record'].unique()

# Processa cada registro
for reg in tqdm(registros_unicos, desc="Segmentando registros"):
    print(f"\nProcessando registro: {reg} (tipo: {type(reg)})")

    df_reg = df[df['record'] == reg]
    if df_reg.empty:
        print(f"Nenhum batimento encontrado para o registro {reg}.")
        continue

    try:
        # Tenta carregar o registro
        record_path = os.path.join(local_path, str(reg))
        record = wfdb.rdrecord(record_path)

        print(f"Registro {reg} carregado. Formato: {record.p_signal.shape}, Canais: {record.sig_name}")

        # Verifica canal MLII
        if 'MLII' in record.sig_name:
            idx_mlII = record.sig_name.index('MLII')
            print(f"Canal MLII encontrado no índice {idx_mlII}")
        else:
            print(f"Registro {reg} não contém canal MLII. Pulando.")
            continue

        # Carrega apenas MLII
        record = wfdb.rdrecord(record_path, channels=[idx_mlII])
        sinal = record.p_signal[:, 0]
        print(f"Sinal MLII com {len(sinal)} amostras.")

    except Exception as e:
        print(f"Erro ao carregar {reg}: {e}")
        continue

    segmentos = []
    rotulos = []

    for i, (_, row) in enumerate(df_reg.iterrows()):
        sample = int(row['sample'])
        classe = int(row['class'])

        inicio = sample - window_left
        fim = sample + window_right

        # Ignora batimentos que extrapolam os limites do sinal
        if inicio < 0 or fim > len(sinal):
            print(f"Ignorando batimento {i} no registro {reg}: janela fora dos limites.")
            continue

        segmento = sinal[inicio:fim]

        # Normalização para o intervalo [0, 1]
        min_val = np.min(segmento)
        max_val = np.max(segmento)

        # Evita divisão por zero em sinais constantes
        if max_val - min_val == 0:
            print(f"Batimento {i} com sinal constante. Ignorado.")
            continue

        segmento_normalizado = (segmento - min_val) / (max_val - min_val)

        if i < 3:  # Mostra os 3 primeiros
            print(f"Sample {sample}, Classe {classe}, Janela: [{inicio}:{fim}]")
            #print(f"Segmento normalizado (5 primeiras amostras): {segmento_normalizado[:5]}")

        segmentos.append(segmento_normalizado.astype('float32'))
        rotulos.append(classe)

    if segmentos:
        segmentos_array = np.array(segmentos, dtype='float32')
        rotulos_array = np.array(rotulos, dtype='int32')

    # Diretório específico para o registro
        base_out_path = '/home/joaovfg/PFC-WPW/mit-bih-segmented-signals/'
        reg_out_path = os.path.join(base_out_path, str(reg))
        os.makedirs(reg_out_path, exist_ok=True)

    # Salva arquivos individuais por segmento
    for idx, (seg, rot) in enumerate(zip(segmentos, rotulos)):
        seg_filename = f'segment_{idx:04d}.dat'
        label_filename = f'label_{idx:04d}.npy'

        seg_path = os.path.join(reg_out_path, seg_filename)
        label_path = os.path.join(reg_out_path, label_filename)

        seg.astype('float32').tofile(seg_path)
        np.save(label_path, np.array(rot, dtype='int32'))

    print(f"{reg}: {len(segmentos)} segmentos salvos  em {reg_out_path}")
