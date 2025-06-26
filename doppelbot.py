import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import readline  # Melhora a experiência de input no terminal
import sys       # Para ler argumentos da linha de comando
import os        # Para verificar se o arquivo existe

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

# --- Carrega o template do prompt do arquivo ---
prompt_template = carregar_prompt_base(caminho_descricao)

# --- Carrega modelo com LoRA ---
print("Carregando modelo e tokenizador...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapters_path)
model = model.eval() # Coloca o modelo em modo de avaliação

tokenizer = AutoTokenizer.from_pretrained(adapters_path)

# --- Monta o prompt de sistema com base na categoria ---
categoria = input("Com quem o Doppelbot está falando? (ex: amigo, interesse romântico...): ").strip()
if not categoria:
    categoria = "amigo" # Define um padrão

system_prompt = prompt_template.format(categoria=categoria)

# --- Loop de chat ---
# A lista `historico_chat` vai guardar a memória da conversa
historico_chat = [{"role": "system", "content": system_prompt}]

print("\n=== Chat com Doppelbot iniciado ===")
print("(Digite 'sair' para encerrar ou 'limpar' para reiniciar o histórico)\n")

while True:
    user_input = input("Você: ").strip()
    if user_input.lower() in ["sair", "exit", "quit"]:
        break
    if user_input.lower() == 'limpar':
        historico_chat = [{"role": "system", "content": system_prompt}]
        print("\n--- Histórico da conversa limpo ---\n")
        continue

    # Adiciona a nova mensagem do usuário ao histórico
    historico_chat.append({"role": "user", "content": user_input})

    # Usa o 'apply_chat_template' para formatar todo o histórico corretamente
    input_ids = tokenizer.apply_chat_template(
        historico_chat,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Gera a resposta
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.eos_token_id
        )

    # Decodifica apenas a parte nova da resposta
    resposta_ids = outputs[0][input_ids.shape[-1]:]
    resposta_texto = tokenizer.decode(resposta_ids, skip_special_tokens=True).strip()

    print(f"Doppelbot: {resposta_texto}\n")

    # Adiciona a resposta do bot ao histórico para o próximo turno
    historico_chat.append({"role": "assistant", "content": resposta_texto})

    # Limita o histórico para não estourar a memória (guarda os últimos 8 turnos)
    if len(historico_chat) > (1 + 8 * 2): # 1 system + 8 pares de user/assistant
        historico_chat = [historico_chat[0]] + historico_chat[-16:]