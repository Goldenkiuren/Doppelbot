import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from peft import PeftModel
import sys
import os
import json
import random
import numpy as np
import pandas as pd
import re

# --- Validação e Configuração Inicial ---
if len(sys.argv) != 2:
    print("Uso: python analise_quantitativa.py <N_amostras>")
    print("Exemplo: python analise_quantitativa.py 150")
    sys.exit(1)

try:
    N_SAMPLES = int(sys.argv[1])
except ValueError:
    print("ERRO: O número de amostras deve ser um inteiro.")
    sys.exit(1)

# --- Caminhos e IDs de Modelo ---
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTERS_PATH = "doppelbot-llama3-8b-instruct-adapters"
DATASET_FILE = "dataset_final.jsonl"
OUTPUT_FILE = "analise_resultados.txt" # Nome do arquivo de saída
EMBEDDING_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2' # Modelo leve e eficiente para embeddings

# --- Funções de Carregamento ---

def carregar_doppelbot():
    """Carrega o modelo Doppelbot fine-tuned com LoRA."""
    print("Carregando modelo Doppelbot e tokenizador...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTERS_PATH)
    model = PeftModel.from_pretrained(model, ADAPTERS_PATH)
    model = model.eval()
    print("Modelo Doppelbot carregado.")
    return model, tokenizer

def carregar_dados_amostra(filepath, n):
    """Carrega o dataset e seleciona N amostras aleatórias."""
    if not os.path.exists(filepath):
        print(f"ERRO: Arquivo de dataset '{filepath}' não encontrado.")
        sys.exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    if len(data) < n:
        print(f"AVISO: O dataset tem apenas {len(data)} amostras. Usando todas as amostras.")
        return data
    
    return random.sample(data, n)

# --- Função de Geração de Resposta ---

def gerar_respostas_bot(model, tokenizer, prompts):
    """Gera respostas do Doppelbot para uma lista de prompts."""
    respostas_bot = []
    print(f"\nGerando {len(prompts)} respostas do Doppelbot...")
    for i, prompt_data in enumerate(prompts):
        print(f"Processando prompt {i+1}/{len(prompts)}...")
        
        # Usa o mesmo formato de prompt que o fine-tuning espera
        conversa = [
            {"role": "system", "content": prompt_data.get("system", "")}, # Usa o system prompt do dataset se houver
            {"role": "user", "content": prompt_data["input"]}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            conversa,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.25,
                    no_repeat_ngram_size=3,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=tokenizer.eos_token_id
                )
        
        resposta_ids = outputs[0][input_ids.shape[-1]:]
        resposta_texto = tokenizer.decode(resposta_ids, skip_special_tokens=True).strip()
        respostas_bot.append(resposta_texto)
        
    print("Geração de respostas concluída.")
    return respostas_bot

# --- Funções de Cálculo de Métricas ---

def calcular_metricas_quantitativas(textos):
    """Calcula um conjunto de métricas quantitativas para uma lista de textos."""
    if not textos:
        return {}
        
    num_palavras_lista = [len(re.findall(r'\w+', texto)) for texto in textos]
    num_linhas_lista = [texto.count('\n') + 1 for texto in textos]
    
    comprimentos_linhas = []
    for texto in textos:
        linhas = texto.split('\n')
        for linha in linhas:
            if linha.strip():
                comprimentos_linhas.append(len(re.findall(r'\w+', linha)))

    todas_palavras = re.findall(r'\w+', ' '.join(textos).lower())
    palavras_unicas = set(todas_palavras)
    ttr = len(palavras_unicas) / len(todas_palavras) if todas_palavras else 0

    return {
        "Tamanho Médio da Resposta (palavras)": np.mean(num_palavras_lista),
        "Desvio Padrão (Tamanho da Resposta)": np.std(num_palavras_lista),
        "Número Médio de Linhas": np.mean(num_linhas_lista),
        "Desvio Padrão (Número de Linhas)": np.std(num_linhas_lista),
        "Comprimento Médio de Linha (palavras)": np.mean(comprimentos_linhas) if comprimentos_linhas else 0,
        "Riqueza Lexical (TTR)": ttr
    }

def calcular_similaridade_semantica(respostas_humanas, respostas_bot, model_id):
    """Calcula a similaridade de cosseno média entre duas listas de textos."""
    print("\nCarregando modelo de embeddings para análise semântica...")
    model = SentenceTransformer(model_id)
    print("Modelo de embeddings carregado.")

    print("Calculando embeddings...")
    embeddings_humanas = model.encode(respostas_humanas)
    embeddings_bot = model.encode(respostas_bot)
    
    similaridades = [cosine_similarity([emb_hum], [emb_bot])[0][0] for emb_hum, emb_bot in zip(embeddings_humanas, embeddings_bot)]
    
    return np.mean(similaridades)

# --- Bloco Principal de Execução ---

if __name__ == "__main__":
    # 1. Carregar dados
    amostras = carregar_dados_amostra(DATASET_FILE, N_SAMPLES)
    prompts = [{"input": item["input"], "system": item.get("system", "")} for item in amostras]
    respostas_humanas = [item["output"] for item in amostras]

    # 2. Carregar modelo e gerar respostas
    doppelbot_model, doppelbot_tokenizer = carregar_doppelbot()
    respostas_bot = gerar_respostas_bot(doppelbot_model, doppelbot_tokenizer, prompts)
    
    # 3. Calcular métricas
    print("\nCalculando métricas para os textos...")
    metricas_humanas = calcular_metricas_quantitativas(respostas_humanas)
    metricas_bot = calcular_metricas_quantitativas(respostas_bot)
    
    similaridade_media = calcular_similaridade_semantica(respostas_humanas, respostas_bot, EMBEDDING_MODEL_ID)
    
    # 4. Apresentar resultados em um arquivo .txt
    print(f"\nSalvando resultados em '{OUTPUT_FILE}'...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("--- ANÁLISE QUANTITATIVA E SEMÂNTICA COMPARATIVA ---\n\n")
        
        # Criar um DataFrame do Pandas
        df_data = {
            "Métrica": list(metricas_humanas.keys()),
            "Humano (Original)": list(metricas_humanas.values()),
            "Doppelbot (Gerado)": list(metricas_bot.values())
        }
        df = pd.DataFrame(df_data)
        df = df.round(4)
        
        # Escrever o DataFrame formatado no arquivo
        f.write(df.to_string(index=False))
        f.write("\n\n") # Adiciona espaço entre as seções
        
        f.write("--- ANÁLISE DE SIMILARIDADE ---\n\n")
        f.write(f"Similaridade de Cosseno Média (Humano vs. Bot): {similaridade_media:.4f}\n\n")
        f.write("(Valores de similaridade mais próximos de 1.0 indicam maior semelhança de significado.)\n")
    
    print("\nAnálise concluída.")