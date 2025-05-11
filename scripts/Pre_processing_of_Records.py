import wfdb
from wfdb import rdrecord
import os
import pandas as pd

print("Leitura e Pré-processamento dos Registros\n")

# Lista de registros da base MIT-BIH
registros = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124', '200', '201', '202', '203', '205', '207',
    '208', '209', '210', '212', '213', '214', '215', '217', '219', '220',
    '221', '222', '223', '228', '230', '231', '232', '233', '234'
]

# Diretório onde os arquivos foram baixados, ou deixar vazio para baixar automaticamente
local_path = '/home/joaovfg/PFC-WPW/mit-bih-arrhythmia-database-1.0.0/'

# Itera sobre os registros e imprime os canais disponíveis
for reg in registros:
    try:
        record = rdrecord(os.path.join(local_path, reg))
        canais = record.sig_name
        print(f'Registro {reg}: Canais disponíveis -> {canais}')
    except Exception as e:
        print(f'Erro ao carregar registro {reg}: {e}')


# Diretório onde os registros estão salvos (ou vazio se forem baixados automaticamente)
local_path = '/home/joaovfg/PFC-WPW/mit-bih-arrhythmia-database-1.0.0/'

# Armazenar registros válidos com MLII
registros_com_mlII = {}

for reg in registros:
    try:
        # Carrega o cabeçalho para verificar os canais disponíveis
        record = rdrecord(os.path.join(local_path, reg), channels=None)
        canais = record.sig_name

        if 'MLII' not in canais:
            print(f"Registro {reg} ignorado: canal MLII não encontrado.")
            continue

        # Índice do canal MLII
        idx_mlII = canais.index('MLII')

        # Recarrega o registro apenas com o canal MLII
        record_mlII = rdrecord(os.path.join(local_path, reg), channels=[idx_mlII])

        registros_com_mlII[reg] = record_mlII
        print(f"Registro {reg} carregado com canal MLII (índice {idx_mlII}).")

    except Exception as e:
        print(f"Erro ao processar registro {reg}: {e}")


print("\nExtração de Batimentos e Classes\n" \
"")

# Rótulos de interesse
nao_wpw_labels = ['N', 'R', 'L', 'f', 'F', '/', 'V', 'A', 'a', 'j']
wpw_label = '*'

# Armazena batimentos filtrados
batimentos = []

for reg, record in registros_com_mlII.items():
    try:
        # Tenta carregar anotações modificadas (com _modified)
        atr_path = os.path.join(local_path, f"{reg}_modified")
        if not os.path.exists(f"{atr_path}.atr"):
            # Se não houver modificado, usa o padrão
            atr_path = os.path.join(local_path, reg)

        annotation = wfdb.rdann(atr_path, 'atr')

        for idx, symbol in enumerate(annotation.symbol):
            if symbol in nao_wpw_labels:
                batimentos.append({
                    'record': reg,
                    'sample': annotation.sample[idx],
                    'symbol': symbol,
                    'class': 0  # Não WPW
                })
            elif symbol == wpw_label:
                batimentos.append({
                    'record': reg,
                    'sample': annotation.sample[idx],
                    'symbol': symbol,
                    'class': 1  # WPW
                })

        print(f"Registro {reg}: {len(annotation.sample)} anotações lidas, {len([b for b in batimentos if b['record'] == reg])} relevantes.")

    except Exception as e:
        print(f"Erro ao ler anotações do registro {reg}: {e}")


# Adiciona a coluna 'channel'
for b in batimentos:
    b['channel'] = 'MLII'

# Cria o DataFrame
df_batimentos = pd.DataFrame(batimentos, columns=['record', 'sample', 'symbol', 'channel', 'class'])

# Salvar o DataFrame em CSV
df_batimentos.to_csv('/home/joaovfg/PFC-WPW/dataFrames/batimentos_filtrados.csv', index=False)
print("Arquivo 'batimentos_filtrados.csv' salvo com sucesso.")

