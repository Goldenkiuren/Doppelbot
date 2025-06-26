import json
import os
import sys

# --- ARQUIVOS DE ENTRADA E SAÍDA ---
ARQUIVO_ENTRADA = "dataset_final.jsonl"
ARQUIVO_SAIDA = "dataset_instruct.jsonl"

def carregar_template_de_arquivo(caminho_arquivo):
    """
    Carrega o template do prompt de sistema a partir de um arquivo de texto.
    """
    if not os.path.exists(caminho_arquivo):
        print(f"ERRO FATAL: Arquivo de descrição '{caminho_arquivo}' não encontrado.")
        sys.exit(1)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"ERRO FATAL: Falha ao ler o arquivo de descrição '{caminho_arquivo}': {e}")
        sys.exit(1)

def criar_dataset_com_instrucoes(system_prompt_template):
    """
    Lê o dataset JSONL, formata cada entrada com o template de prompt do Llama 3
    (system, user, assistant) e salva em um novo arquivo.
    """
    if not os.path.exists(ARQUIVO_ENTRADA):
        print(f"ERRO: Arquivo de entrada '{ARQUIVO_ENTRADA}' não encontrado.")
        print("Certifique-se de ter executado as etapas anteriores do pipeline.")
        sys.exit(1)

    print(f"Iniciando a criação do dataset instrucional a partir de '{ARQUIVO_ENTRADA}'...")

    dados_formatados = []
    
    with open(ARQUIVO_ENTRADA, 'r', encoding='utf-8') as f_in:
        for linha in f_in:
            try:
                exemplo = json.loads(linha)
                
                input_original = exemplo.get("input")
                output_original = exemplo.get("output")
                categoria = exemplo.get("categoria", "desconhecido")

                if not input_original or not output_original:
                    continue

                prompt_sistema = system_prompt_template.format(categoria=categoria)

                conversa_formatada = [
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": input_original},
                    {"role": "assistant", "content": output_original}
                ]
                
                dados_formatados.append(conversa_formatada)

            except json.JSONDecodeError:
                print(f"AVISO: Pulando linha mal formatada: {linha.strip()}")
                continue
    
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f_out:
        for item in dados_formatados:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("-" * 50)
    print("Etapa de injeção de instrução concluída com sucesso!")
    print(f"Foram criados {len(dados_formatados)} exemplos de treino.")
    print(f"O novo dataset instrucional foi salvo em: '{ARQUIVO_SAIDA}'")
    print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python add_instruction.py \"caminho/para/descricao.txt\"")
        sys.exit(1)
    
    caminho_descricao = sys.argv[1]
    template = carregar_template_de_arquivo(caminho_descricao)
    criar_dataset_com_instrucoes(template)