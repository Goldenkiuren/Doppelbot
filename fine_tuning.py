import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os
import json

# --- 1. Configurações ---
MODELO_BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
NOME_DATASET = "dataset_instruct.jsonl"
NOME_NOVO_MODELO = "doppelbot-llama3-8b-instruct-adapters"

# --- Validação de Arquivo ---
if not os.path.exists(NOME_DATASET):
    raise FileNotFoundError(
        f"ERRO: O arquivo de dataset '{NOME_DATASET}' não foi encontrado. "
        "Execute as etapas anteriores do pipeline primeiro."
    )

# --- 2. Configuração de Quantização (QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 3. Carregamento do Modelo e Tokenizador ---
print(f"Carregando modelo base: {MODELO_BASE}")
model = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. Configuração do LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# --- 5. Carregamento e Processamento do Dataset ---
print(f"Carregando e processando dataset: {NOME_DATASET}")
dataset = load_dataset("text", data_files={"train": NOME_DATASET}, split="train")

def formatar_para_chat(linha):
    conversa = json.loads(linha["text"])
    mensagens_formatadas = ""
    for msg in conversa:
        if msg["role"] == "user":
            mensagens_formatadas += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            mensagens_formatadas += f"<|assistant|>\n{msg['content']}\n"
        else:
            mensagens_formatadas += f"<|{msg['role']}|>\n{msg['content']}\n"
    return {"messages": mensagens_formatadas.strip()}

dataset = dataset.map(formatar_para_chat, remove_columns=["text"])

# --- 6. Configuração do SFT (substitui TrainingArguments) ---
sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to=None,
    dataset_text_field="messages",
    max_seq_length=512,
    packing=False,
    save_steps=50,
    logging_steps=10,
)


# --- 7. Inicialização do Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
)

# --- 8. Iniciar Treinamento ---
print("Iniciando o processo de fine-tuning com QLoRA...")
trainer.train()
print("Fine-tuning concluído!")

# --- 9. Salvar os Adaptadores Treinados ---
print(f"Salvando adaptadores do modelo em '{NOME_NOVO_MODELO}'...")
trainer.save_model(NOME_NOVO_MODELO)
tokenizer.save_pretrained(NOME_NOVO_MODELO)
print("Processo finalizado com sucesso!")
