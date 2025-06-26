#!/bin/bash

# Este script automatiza o pipeline completo: dados, formatação e fine-tuning.
# Etapa 1: Normaliza e anonimiza os nomes.
# Etapa 2: Gera o dataset base (input/output).
# Etapa 3: Injeta a persona/instrução no dataset.
# Etapa 4: Executa o fine-tuning com QLoRA.

# --- VALIDAÇÃO DE ENTRADA ---
if [ "$#" -ne 3 ]; then
    echo "ERRO: Número incorreto de parâmetros."
    echo "Uso: ./run_pipeline.sh \"Seu Nome\" \"pasta_de_origem\" \"arquivo_de_descricao\""
    echo "Exemplo: ./run_pipeline.sh \"Augusto\" \"conversas_originais\" \"descricao.txt\""
    exit 1
fi

USER_NAME="$1"
SOURCE_FOLDER="$2"
DESCRIPTION_FILE="$3"

# Validações de arquivos e pastas
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "ERRO: A pasta de origem '$SOURCE_FOLDER' não foi encontrada."
    exit 1
fi
if [ ! -f "$DESCRIPTION_FILE" ]; then
    echo "ERRO: O arquivo de descrição '$DESCRIPTION_FILE' não foi encontrado."
    exit 1
fi

# --- EXECUÇÃO DO PIPELINE ---
echo "================================================="
echo "  PIPELINE COMPLETO DOPPELBOT"
echo "================================================="
echo "Usuário a ser imitado: $USER_NAME"
echo "Pasta de origem: $SOURCE_FOLDER"
echo "Arquivo de Persona: $DESCRIPTION_FILE"
echo "-------------------------------------------------"

# Ativar ambiente virtual (recomendado)
# source venv/bin/activate

# Etapa 1: Normalização
echo ""
echo ">>> ETAPA 1: Normalizando nomes e anonimizando..."
python3 name_normalize.py "$USER_NAME" "$SOURCE_FOLDER" || { echo "ERRO FATAL na Etapa 1."; exit 1; }

# Etapa 2: Pré-processamento
echo ""
echo ">>> ETAPA 2: Gerando dataset base 'dataset_final.jsonl'..."
python3 pre_processing.py || { echo "ERRO FATAL na Etapa 2."; exit 1; }

# Etapa 3: Injeção da Instrução
echo ""
echo ">>> ETAPA 3: Injetando persona em 'dataset_instruct.jsonl'..."
python3 add_instruction.py "$DESCRIPTION_FILE" || { echo "ERRO FATAL na Etapa 3."; exit 1; }

# Etapa 4: Fine-tuning
echo ""
echo ">>> ETAPA 4: INICIANDO FINE-TUNING DO MODELO. ESTA ETAPA PODE LEVAR HORAS."
python3 fine_tuning.py || { echo "ERRO FATAL na Etapa 4."; exit 1; }

echo ""
echo "-------------------------------------------------"
echo "PIPELINE CONCLUÍDO COM SUCESSO!"
echo "Os adaptadores do seu Doppelbot foram salvos na pasta 'doppelbot-llama3-8b-instruct-adapters'."
echo "================================================="