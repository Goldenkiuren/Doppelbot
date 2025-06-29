import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import os
import re

# --- Constantes ---
# O prefixo do usuário é mantido para uma comparação justa de estímulos
PREFIXO_PERGUNTA = (
    "Para a afirmação a seguir, descreva como ela se aplica a você. Responda sempre em primeira pessoa (eu).\n\nAfirmação: "
)

# --- Funções Auxiliares ---
def carregar_perguntas(caminho_arquivo):
    """Carrega as perguntas de um arquivo de texto."""
    if not os.path.exists(caminho_arquivo):
        print(f"ERRO: Arquivo de perguntas '{caminho_arquivo}' não encontrado.")
        sys.exit(1)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            return [linha.strip() for linha in f if linha.strip()]
    except Exception as e:
        print(f"ERRO: Falha ao ler o arquivo de perguntas: {e}")
        sys.exit(1)

# --- Validação de Argumentos ---
if len(sys.argv) != 3:
    print("Uso: python avaliar_baseline.py <caminho_perguntas> <caminho_saida>")
    print("Exemplo: python avaliar_baseline.py perguntas_teste.txt respostas_baseline.txt")
    sys.exit(1)

# --- Caminhos e Configurações ---
caminho_perguntas = sys.argv[1]
caminho_saida = sys.argv[2]
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Carregamento de Dados ---
perguntas = carregar_perguntas(caminho_perguntas)
print(f"Encontradas {len(perguntas)} perguntas em '{caminho_perguntas}'.")

# --- Carregamento do Modelo BASE (sem adaptadores LoRA) ---
print("Carregando modelo e tokenizador BASE...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Modelo BASE carregado. Nenhum adaptador LoRA foi aplicado.")
model = model.eval()


# --- Processamento das Perguntas e Geração das Respostas ---
print(f"\nIniciando geração de respostas... Os resultados serão salvos em '{caminho_saida}'.")

try:
    with open(caminho_saida, 'w', encoding='utf-8') as f_out:
        for i, pergunta_texto in enumerate(perguntas):
            num_pergunta = i + 1
            print(f"Processando pergunta {num_pergunta}/{len(perguntas)}...")

            # Monta o prompt do usuário com o prefixo e a pergunta
            user_prompt_final = f"{PREFIXO_PERGUNTA}{pergunta_texto}"

            # Prepara a conversa SEM NENHUM system prompt
            conversa_atual = [
                {"role": "user", "content": user_prompt_final}
            ]

            input_ids = tokenizer.apply_chat_template(
                conversa_atual,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # Gera a resposta usando os mesmos hiperparâmetros para uma comparação justa
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.25,
                    no_repeat_ngram_size=3,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decodifica e formata a resposta
            resposta_ids = outputs[0][input_ids.shape[-1]:]
            resposta_bruta = tokenizer.decode(resposta_ids, skip_special_tokens=True).strip()

            # Escreve no arquivo de saída
            f_out.write(f"--- Pergunta {num_pergunta} ---\n")
            f_out.write(f"Texto: {pergunta_texto}\n")
            f_out.write(f"Resposta do Modelo Base: {resposta_bruta}\n\n")

    print("\nProcesso concluído com sucesso!")
    print(f"As respostas do modelo BASE foram salvas em '{caminho_saida}'.")

except Exception as e:
    print(f"\nERRO: Ocorreu uma falha durante a geração das respostas: {e}")
    sys.exit(1)