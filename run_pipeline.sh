#!/bin/bash

# Este script automatiza o pipeline completo de pré-processamento e formatação.
# Etapa 1: Normaliza e anonimiza os nomes.
# Etapa 2: Gera o dataset base (input/output).
# Etapa 3: Injeta a persona/instrução no dataset final.

# --- VALIDAÇÃO DE ENTRADA ---
if [ "$#" -ne 3 ]; then
    echo "ERRO: Número incorreto de parâmetros."
    echo ""
    echo "Uso: ./run_pipeline.sh \"Seu Nome no WhatsApp\" \"pasta_de_origem\" \"arquivo_de_descricao\""
    echo "Exemplo: ./run_pipeline.sh \"Augusto\" \"conversas_originais\" \"descricao.txt\""
    exit 1
fi

USER_NAME="$1"
SOURCE_FOLDER="$2"
DESCRIPTION_FILE="$3"

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
echo "  PIPELINE DE DADOS DOPPELBOT"
echo "================================================="
echo "Usuário a ser imitado: $USER_NAME"
echo "Pasta de origem: $SOURCE_FOLDER"
echo "Arquivo de Persona: $DESCRIPTION_FILE"
echo "-------------------------------------------------"

# Etapa 1: Normalização e Anonimização
echo ""
echo ">>> ETAPA 1: Normalizando nomes e anonimizando..."
python3 name_normalize.py "$USER_NAME" "$SOURCE_FOLDER"
if [ $? -ne 0 ]; then
    echo "ERRO FATAL na Etapa 1. Abortando."
    exit 1
fi


# Etapa 2: Pré-processamento e Geração do Dataset
echo ""
echo ">>> ETAPA 2: Gerando dataset base 'dataset_final.jsonl'..."
python3 pre_processing.py
if [ $? -ne 0 ]; then
    echo "ERRO FATAL na Etapa 2. Abortando."
    exit 1
fi

# Etapa 3: Injeção da Instrução de Persona
echo ""
echo ">>> ETAPA 3: Injetando persona e criando 'dataset_instruct.jsonl'..."
python3 add_instruction.py "$DESCRIPTION_FILE"
if [ $? -ne 0 ]; then
    echo "ERRO FATAL na Etapa 3. Abortando."
    exit 1
fi


echo ""
echo "-------------------------------------------------"
echo "PIPELINE CONCLUÍDO COM SUCESSO!"
echo "O dataset final para fine-tuning foi salvo em 'dataset_instruct.jsonl'."
echo "================================================="