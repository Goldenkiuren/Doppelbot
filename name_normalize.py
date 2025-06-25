import os
import re
import sys

# --- CONSTANTES DE CONFIGURAÇÃO ---

# Nome da pasta onde os novos arquivos padronizados serão criados.
PASTA_PADRONIZADA = "conversas_padronizadas"

# O prefixo padrão dos arquivos de conversa do WhatsApp.
PREFIXO_NOME_ARQUIVO = "Conversa do WhatsApp com "

def padronizar_conversas(meu_nome, pasta_originais):
    """
    Lê arquivos de conversas de uma estrutura de pastas categorizadas,
    substitui os nomes reais por rótulos genéricos e salva os novos arquivos.
    """
    if not os.path.isdir(pasta_originais):
        print(f"ERRO: A pasta de origem '{pasta_originais}' não foi encontrada.")
        sys.exit(1)

    if not os.path.exists(PASTA_PADRONIZADA):
        os.makedirs(PASTA_PADRONIZADA)
        print(f"Pasta de destino '{PASTA_PADRONIZADA}' criada.")

    print("Iniciando processo de padronização e anonimização...")
    print("-" * 50)

    for categoria in os.listdir(pasta_originais):
        pasta_categoria_origem = os.path.join(pasta_originais, categoria)
        if os.path.isdir(pasta_categoria_origem):
            pasta_categoria_destino = os.path.join(PASTA_PADRONIZADA, categoria)
            if not os.path.exists(pasta_categoria_destino):
                os.makedirs(pasta_categoria_destino)

            print(f"\nProcessando categoria: '{categoria}'")
            contador_categoria = 1

            for nome_arquivo in sorted(os.listdir(pasta_categoria_origem)):
                if nome_arquivo.endswith(".txt") and nome_arquivo.startswith(PREFIXO_NOME_ARQUIVO):
                    caminho_arquivo_original = os.path.join(pasta_categoria_origem, nome_arquivo)
                    try:
                        nome_interlocutor = nome_arquivo.replace(PREFIXO_NOME_ARQUIVO, "").replace(".txt", "")
                        novo_rotulo = f"{categoria}{contador_categoria}"

                        with open(caminho_arquivo_original, 'r', encoding='utf-8') as f_origem:
                            conteudo = f_origem.read()

                        conteudo_modificado = re.sub(rf'({re.escape(nome_interlocutor)}):', rf'{novo_rotulo}:', conteudo)
                        conteudo_modificado = re.sub(rf'({re.escape(meu_nome)}):', r'MeuNome:', conteudo_modificado)

                        novo_nome_arquivo = f"{novo_rotulo}.txt"
                        caminho_arquivo_novo = os.path.join(pasta_categoria_destino, novo_nome_arquivo)

                        with open(caminho_arquivo_novo, 'w', encoding='utf-8') as f_destino:
                            f_destino.write(conteudo_modificado)

                        print(f"  - '{nome_arquivo}' -> '{novo_nome_arquivo}'")
                        contador_categoria += 1
                    except Exception as e:
                        print(f"  - [ERRO] Falha ao processar o arquivo '{nome_arquivo}': {e}")

    print("-" * 50)
    print("Etapa 1 (Padronização) concluída com sucesso!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python name_normalize.py \"Seu Nome\" \"caminho/para/conversas_originais\"")
        sys.exit(1)
    
    MEU_NOME_ARG = sys.argv[1]
    PASTA_ORIGINAIS_ARG = sys.argv[2]
    
    padronizar_conversas(MEU_NOME_ARG, PASTA_ORIGINAIS_ARG)
