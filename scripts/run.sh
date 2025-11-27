#!/bin/bash

# Script para executar o programa principal com cada dataset para todos os tipos de floating point
# Uso: ./run.sh
#
# Will execute the build/cnn_experiment program:
# Usage: {run.sh_dir}/../build/cnn_experiment --dataset Fruits360|PKLot [--data_root DIR] [--epochs N] [--batch_size N]
#              [--lr LR] [--dtype float64|float32|float16|bfloat16] [--seed S]
#              [--use_dropout] [--max_ram_mb MB] [--load_model PATH] [--load_state PATH]

# Options:
#   --dataset         Dataset name (Fruits360 or PKLot) [required]
#   --data_root      Root directory for datasets (default: ./data/datasets)
#   --epochs         Number of training epochs (default: 10)
#   --batch_size     Mini-batch size (default: 512)
#   --lr             Learning rate (default: 1e-3)
#   --dtype          Data type: float64, float32, float16, bfloat16 (default: float32)
#   --seed           Random seed (default: 42)
#   --use_dropout    Use dropout layer before fully connected layers
#   --max_ram_mb     Max RAM in MB before using memmap fallback (default: 2048)
#   --load_model     Path to load full TorchScript Module
#   --load_state     Path to load state_dict saved from Python
#   --cpu            Force using CPU even if GPU is available
#   --model_dir      Directory to save trained model and metrics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
PROGRAM="${BUILD_DIR}/cnn_experiment"
DATASETS=("Fruits360" "PKLot")
DTYPES=( "float32" "float16" "bfloat16" "float64")
BATCH_SIZES=(256 512 1024 2048) # Tamanhos para rodar com cada dtype (cada dtype deve rodar todos os batch sizes)

# Seed base e número de repetições
SEED=42
RUNS=5

LOGS_DIR="${SCRIPT_DIR}/logs" # Salvar logs de saída para todos os stdout e stderr e informacoes de erro de execução
mkdir -p "${LOGS_DIR}"

# Crie um diretório para a atual execucao com o timestamp
LOG_DIR="${LOGS_DIR}/run_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "${LOG_DIR}"

# Cada modelo deve ser salvo em um diretório específico com base no dataset, dtype, batch size e seed
# Exemplo: models/Fruits360/float32_bs512_seed42/

for DATASET in "${DATASETS[@]}"; do
  for DTYPE in "${DTYPES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for RUN_IDX in $(seq 0 $((RUNS - 1))); do
        seed=$((SEED + RUN_IDX))

        MODEL_DIR="${SCRIPT_DIR}/models/${DATASET}/${DTYPE}_bs${BATCH_SIZE}_seed${seed}"
        mkdir -p "${MODEL_DIR}"

        LOG_FILE="${LOG_DIR}/run_${DATASET}_${DTYPE}_bs${BATCH_SIZE}_seed${seed}.log"

        echo "Executando ${PROGRAM} com dataset=${DATASET}, dtype=${DTYPE}, batch_size=${BATCH_SIZE}, seed=${seed}"
        echo "Logs serão salvos em ${LOG_FILE}"
        
        # Executar o programa e redirecionar stdout e stderr para o arquivo de log
        {
            "${PROGRAM}" --dataset "${DATASET}" --batch_size "${BATCH_SIZE}" --dtype "${DTYPE}" --model_dir "${MODEL_DIR}" --seed "${seed}"
        } |& tee "${LOG_FILE}"

        if [ $? -ne 0 ]; then
          echo "Erro ao executar ${PROGRAM} com dataset=${DATASET}, dtype=${DTYPE}, batch_size=${BATCH_SIZE}, seed=${seed}. Verifique o log em ${LOG_FILE} para mais detalhes."
        else
          echo "Execução concluída com sucesso para dataset=${DATASET}, dtype=${DTYPE}, batch_size=${BATCH_SIZE}, seed=${seed}."
        fi
      done
    done
  done
done
