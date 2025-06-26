# analisar_dataset.py (v2 - com gráfico melhorado)
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os

# --- CONFIGURAÇÕES ---
NOME_DATASET = "dataset_instruct.jsonl"
NOME_MODELO = "meta-llama/Meta-Llama-3-8B-Instruct"
ARQUIVO_GRAFICO = "distribuicao_tokens_zoom.png" # Novo nome para o arquivo do gráfico

def analisar_comprimento_sequencias():
    if not os.path.exists(NOME_DATASET):
        print(f"ERRO: Arquivo de dataset '{NOME_DATASET}' não encontrado.")
        return

    print(f"Carregando o tokenizer de '{NOME_MODELO}'...")
    tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO)

    print(f"Processando o dataset '{NOME_DATASET}' para contar os tokens...")
    comprimentos_tokens = []
    with open(NOME_DATASET, 'r', encoding='utf-8') as f:
        for linha in f:
            try:
                conversa = json.loads(linha)
                token_ids = tokenizer.apply_chat_template(conversa, add_generation_prompt=False)
                comprimentos_tokens.append(len(token_ids))
            except json.JSONDecodeError:
                continue

    if not comprimentos_tokens:
        print("Nenhum dado válido encontrado para análise.")
        return

    # --- CÁLCULO DAS ESTATÍSTICAS ---
    max_len = int(np.max(comprimentos_tokens))
    avg_len = int(np.mean(comprimentos_tokens))
    median_len = int(np.median(comprimentos_tokens))
    p90 = int(np.percentile(comprimentos_tokens, 90))
    p95 = int(np.percentile(comprimentos_tokens, 95))
    p99 = int(np.percentile(comprimentos_tokens, 99))
    p99_5 = int(np.percentile(comprimentos_tokens, 99.5)) # Usado para limitar o gráfico

    print("\n--- ESTATÍSTICAS DO COMPRIMENTO DAS SEQUÊNCIAS (EM TOKENS) ---")
    print(f"Total de exemplos analisados: {len(comprimentos_tokens)}")
    print(f"Comprimento Máximo (Outlier): {max_len}")
    print(f"Comprimento Médio: {avg_len}")
    print(f"Comprimento Mediano: {median_len}")
    print("-" * 50)
    print("Percentis (muito úteis para definir max_seq_length):")
    print(f"90% dos exemplos têm até: {p90} tokens")
    print(f"95% dos exemplos têm até: {p95} tokens")
    print(f"99% dos exemplos têm até: {p99} tokens")
    print("-" * 50)

    # --- GERAÇÃO DO GRÁFICO MELHORADO ---
    print("Gerando gráfico da distribuição de tokens (com zoom)...")
    plt.figure(figsize=(15, 7))
    # Aumentamos o número de 'bins' para mais detalhes e limitamos o 'range'
    plt.hist(comprimentos_tokens, bins=150, color='deepskyblue', edgecolor='black', range=(0, p99_5))
    
    plt.title(f'Distribuição do Comprimento das Sequências (visão de 0 a {p99_5} tokens)')
    plt.xlabel('Número de Tokens')
    plt.ylabel('Frequência (Número de Exemplos)')

    # Adicionando linhas verticais para os percentis importantes
    plt.axvline(p90, color='green', linestyle='dashed', linewidth=2)
    plt.axvline(p95, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(p99, color='purple', linestyle='dashed', linewidth=2)

    # Adicionando legendas para as linhas
    plt.text(p90, plt.ylim()[1]*0.9, f' 90º Percentil: {p90}', color='green')
    plt.text(p95, plt.ylim()[1]*0.8, f' 95º Percentil: {p95}', color='red')
    plt.text(p99, plt.ylim()[1]*0.7, f' 99º Percentil: {p99}', color='purple')

    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout() # Ajusta o layout para não cortar os textos
    plt.savefig(ARQUIVO_GRAFICO)
    print(f"Gráfico melhorado salvo como '{ARQUIVO_GRAFICO}'.")


if __name__ == "__main__":
    analisar_comprimento_sequencias()