# fine_tuning.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os

# --- 1. Configurações ---
# Modelo base do Hugging Face. O Ollama já o deixou no cache, então o download será rápido.
MODELO_BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
# Dataset com as instruções de persona que criamos.
NOME_DATASET = "dataset_instruct.jsonl"
# Nome final para os adaptadores treinados.
NOME_NOVO_MODELO = "doppelbot-llama3-8b-instruct-adapters"

# --- Validação de Arquivo ---
if not os.path.exists(NOME_DATASET):
    raise FileNotFoundError(f"ERRO: O arquivo de dataset '{NOME_DATASET}' não foi encontrado. "
                            "Execute as etapas anteriores do pipeline primeiro.")

# --- 2. Configuração de Quantização (QLoRA) ---
# Carrega o modelo com precisão de 4 bits para economizar memória.
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
    device_map="auto" # Mapeia o modelo para a GPU automaticamente
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. Configuração do LoRA ---
# Define os parâmetros para os adaptadores que serão treinados.
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[ # Módulos específicos para o Llama 3
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# --- 5. Carregamento do Dataset ---
print(f"Carregando dataset: {NOME_DATASET}")
# O SFTTrainer espera uma coluna com a conversa, por padrão 'text'.
# Como nosso JSONL tem uma lista de dicionários, vamos renomear para 'messages'.
dataset = load_dataset("json", data_files=NOME_DATASET, split="train")
dataset = dataset.rename_column("conversations", "messages") # Renomeia a coluna implícita


# --- 6. Argumentos de Treinamento ---
training_arguments = TrainingArguments(
    output_dir="./results",             # Pasta para checkpoints
    num_train_epochs=1,                 # 1 época é um bom ponto de partida para fine-tuning
    per_device_train_batch_size=4,      # Batch size que deve caber na sua VRAM
    gradient_accumulation_steps=1,      # Aumente se a VRAM for um problema
    optim="paged_adamw_32bit",
    save_steps=50,                      # Salva checkpoints a cada 50 passos
    logging_steps=10,                   # Log de progresso a cada 10 passos
    learning_rate=2e-4,                 # Taxa de aprendizado
    weight_decay=0.001,
    fp16=False,
    bf16=True,                          # Essencial para RTX 40xx
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# --- 7. Inicialização do Trainer ---
# SFTTrainer (Supervised Fine-tuning Trainer) é otimizado para esse tipo de tarefa.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_kwargs={"add_special_tokens": False}, # O template de chat já cuida disso
    max_seq_length=1024,                # Limite do tamanho da sequência
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Desativado para usar o formato de chat
)

# --- 8. Iniciar Treinamento ---
print("Iniciando o processo de fine-tuning com QLoRA...")
trainer.train()
print("Fine-tuning concluído!")

# --- 9. Salvar os Adaptadores Treinados ---
print(f"Salvando adaptadores do modelo em '{NOME_NOVO_MODELO}'...")
trainer.model.save_pretrained(NOME_NOVO_MODELO)
tokenizer.save_pretrained(NOME_NOVO_MODELO)
print("Processo finalizado com sucesso!")