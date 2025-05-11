import wfdb
import numpy as np
import csv

# Carregar sinais e anotações
record = wfdb.rdrecord('230', pn_dir='mitdb')
ann = wfdb.rdann('230', 'atr', pn_dir='mitdb')

# Encontrar índices onde há anotações de ritmo (aux_note não vazio)
rhythm_ann_idx = np.where(np.array(ann.aux_note) != '')[0]

# Lista de pares (início, fim) dos intervalos entre (PREX e (N)
intervalos = []
prex_start = None

for i in rhythm_ann_idx:
    note = ann.aux_note[i]
    if '(PREX' in note:
        prex_start = ann.sample[i]
    elif '(N' in note and prex_start is not None:
        n_end = ann.sample[i]
        intervalos.append((prex_start, n_end))
        prex_start = None  # Resetar para próxima ocorrência

# Identificar que aux_note tem o mesmo tamanho que sample
# Se não, preenche com strings vazias
if len(ann.aux_note) < len(ann.sample):
    ann.aux_note = ann.aux_note + [''] * (len(ann.sample) - len(ann.aux_note))

# Reclassificar batimentos 'N' como '*'
for start, end in intervalos:
    for i in range(len(ann.sample)):
        if start <= ann.sample[i] <= end and ann.symbol[i] == 'N':
            ann.symbol[i] = '*'  # Novo símbolo
            
# Salvar a nova anotação
wfdb.wrann(record_name='230_modified', extension='atr',
           sample=ann.sample, symbol=ann.symbol, aux_note=ann.aux_note)

# Exportar para CSV
csv_filename = '230_modified.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['sample', 'symbol', 'aux_note'])

    for i in range(len(ann.sample)):
        writer.writerow([ann.sample[i], ann.symbol[i], ann.aux_note[i]])
