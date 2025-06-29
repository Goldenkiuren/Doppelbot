import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import os
import re

# --- Constantes ---
PREFIXO_PERGUNTA = (
    "Estou fazendo um teste para avaliar sua personalidade. Para a afirmação abaixo, responda estritamente em primeira pessoa (usando 'Eu'), explicando como ela se aplica ou não a você e ao seu jeito de ser.\n\nAfirmação: "
)

# --- Funções Auxiliares ---
def carregar_arquivo(caminho_arquivo, tipo):
    """Carrega o conteúdo de um arquivo de texto."""
    if not os.path.exists(caminho_arquivo):
        print(f"ERRO: Arquivo de {tipo} '{caminho_arquivo}' não encontrado.")
        sys.exit(1)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            # Retorna uma lista de linhas para o arquivo de perguntas
            if tipo == "perguntas":
                return [linha.strip() for linha in f if linha.strip()]
            # Retorna o conteúdo inteiro para o arquivo de descrição
            return f.read()
    except Exception as e:
        print(f"ERRO: Falha ao ler o arquivo de {tipo}: {e}")
        sys.exit(1)

# --- Validação de Argumentos ---
if len(sys.argv) != 4:
    print("Uso: python avaliar_personalidade.py <caminho_descricao> <caminho_perguntas> <caminho_saida>")
    print("Exemplo: python avaliar_personalidade.py descricao.txt perguntas_teste.txt respostas_bot.txt")
    sys.exit(1)

# --- Caminhos e Configurações ---
caminho_descricao = sys.argv[1]
caminho_perguntas = sys.argv[2]
caminho_saida = sys.argv[3]

base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
adapters_path = "doppelbot-llama3-8b-instruct-adapters"

# --- Carregamento de Dados ---
prompt_template = carregar_arquivo(caminho_descricao, "descrição")
perguntas = carregar_arquivo(caminho_perguntas, "perguntas")
print(f"Encontradas {len(perguntas)} perguntas em '{caminho_perguntas}'.")


# --- Carregamento do Modelo (idêntico ao seu script original) ---
print("Carregando modelo e tokenizador...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 # Adicionado para consistência
)
tokenizer = AutoTokenizer.from_pretrained(adapters_path)

# Ajuste de vocabulário (se necessário)
if "<|msg_sep|>" not in tokenizer.get_vocab():
    print("Adicionando token especial '<|msg_sep|>' ao tokenizador.")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|msg_sep|>']})
    model.resize_token_embeddings(len(tokenizer))

print("Carregando adaptadores LoRA...")
model = PeftModel.from_pretrained(model, adapters_path)
model = model.eval()

# --- Preparação do Prompt de Sistema ---
# Para um teste padronizado, podemos fixar a categoria ou torná-la um argumento extra.
# Fixar como "amigo" é uma escolha razoável para manter a consistência.
categoria = "amigo"
system_prompt = prompt_template.format(categoria=categoria)
print(f"Persona do bot definida para a categoria: '{categoria}'")

# --- Processamento das Perguntas e Geração das Respostas ---
print(f"\nIniciando geração de respostas... Os resultados serão salvos em '{caminho_saida}'.")

try:
    with open(caminho_saida, 'w', encoding='utf-8') as f_out:
        for i, pergunta_texto in enumerate(perguntas):
            num_pergunta = i + 1
            print(f"Processando pergunta {num_pergunta}/{len(perguntas)}...")

            # Monta o prompt do usuário com o prefixo e a pergunta
            user_prompt_final = f"{PREFIXO_PERGUNTA}{pergunta_texto}"

            # Prepara a conversa para o template do chat
            conversa_atual = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_final}
            ]

            input_ids = tokenizer.apply_chat_template(
                conversa_atual,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # Gera a resposta
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=50,  # Reduzido, pois a resposta esperada é curta
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
            f_out.write(f"Resposta do Bot: {resposta_bruta}\n\n")

    print("\nProcesso concluído com sucesso!")
    print(f"As respostas do Doppelbot foram salvas em '{caminho_saida}'.")

except Exception as e:
    print(f"\nERRO: Ocorreu uma falha durante a geração das respostas: {e}")
    sys.exit(1)