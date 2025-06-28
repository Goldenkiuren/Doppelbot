import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import readline
import sys
import os
import re

# --- Função para carregar o prompt de um arquivo ---
def carregar_prompt_base(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        print(f"ERRO: Arquivo de descrição '{caminho_arquivo}' não encontrado.")
        sys.exit(1)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"ERRO: Falha ao ler o arquivo de descrição: {e}")
        sys.exit(1)

# --- Validação de Argumentos ---
if len(sys.argv) != 2:
    print("Uso: python testar_bot.py <caminho_para_o_arquivo_de_descricao>")
    print("Exemplo: python testar_bot.py descricao.txt")
    sys.exit(1)

# --- Caminhos e Configurações ---
caminho_descricao = sys.argv[1]
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
adapters_path = "doppelbot-llama3-8b-instruct-adapters"

prompt_template = carregar_prompt_base(caminho_descricao)

# --- Carrega modelo com LoRA ---
print("Carregando modelo e tokenizador...")
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

tokenizer = AutoTokenizer.from_pretrained(adapters_path)
print("Ajustando vocabulário do modelo para corresponder aos adaptadores...")
if "<|msg_sep|>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|msg_sep|>']})
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, adapters_path)
model = model.eval()

# --- Monta o prompt de sistema com base na categoria ---
categoria = input("Com quem o Doppelbot está falando? (ex: amigo, interesse romântico...): ").strip()
if not categoria:
    categoria = "amigo"

system_prompt = prompt_template.format(categoria=categoria)

print("\n=== Gerador de Respostas Isoladas ===")
print("(Digite 'sair' para encerrar)\n")

while True:
    user_input = input("Você: ").strip()
    if user_input.lower() in ["sair", "exit", "quit"]:
        break

    # O histórico é criado do zero a cada pergunta
    conversa_atual = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Usa o 'apply_chat_template' apenas com a conversa atual, sem histórico
    input_ids = tokenizer.apply_chat_template(
        conversa_atual,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Gera a resposta
    with torch.no_grad():
        outputs = model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=tokenizer.eos_token_id
            )

    # Decodifica apenas a parte nova da resposta
    resposta_ids = outputs[0][input_ids.shape[-1]:]
    resposta_bruta = tokenizer.decode(resposta_ids, skip_special_tokens=True).strip()
    resposta_formatada = re.sub(r'\n{2,}', '\n\n', resposta_bruta)
    
    print(f"Doppelbot:\n{resposta_formatada}\n")