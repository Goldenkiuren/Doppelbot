import os
import re

# --- CONSTANTES DE CONFIGURAÇÃO ---

# O seu nome, exatamente como aparece nos logs do WhatsApp.
MEU_NOME = "Augusto"

# Nome da pasta que conterá os seus arquivos .txt originais, organizados em subpastas.
PASTA_ORIGINAIS = "conversas_originais"

# Nome da pasta onde os novos arquivos padronizados serão criados.
PASTA_PADRONIZADA = "conversas_padronizadas"

# O prefixo padrão dos arquivos de conversa do WhatsApp para extrair o nome do interlocutor.
PREFIXO_NOME_ARQUIVO = "Conversa do WhatsApp com "

# --- SCRIPT DE PADRONIZAÇÃO ---

def padronizar_conversas():
    """
    Este script lê arquivos de conversas de uma estrutura de pastas categorizadas,
    extrai o nome do interlocutor a partir do nome do arquivo, substitui os nomes
    reais por rótulos genéricos e salva os novos arquivos em uma nova estrutura de pastas.
    """
    # Verifica se a pasta de origem existe
    if not os.path.isdir(PASTA_ORIGINAIS):
        print(f"ERRO: A pasta de origem '{PASTA_ORIGINAIS}' não foi encontrada.")
        print("Por favor, crie esta pasta e organize seus arquivos .txt dentro de subpastas de categoria (ex: 'Amigo', 'Amiga').")
        return

    # Cria a pasta de destino principal, se ela não existir
    if not os.path.exists(PASTA_PADRONIZADA):
        os.makedirs(PASTA_PADRONIZADA)
        print(f"Pasta de destino '{PASTA_PADRONIZADA}' criada.")

    print("Iniciando processo de padronização e anonimização de conversas...")
    print("-" * 50)

    # Percorre as subpastas (categorias) dentro da pasta de origem
    for categoria in os.listdir(PASTA_ORIGINAIS):
        pasta_categoria_origem = os.path.join(PASTA_ORIGINAIS, categoria)

        if os.path.isdir(pasta_categoria_origem):
            # Cria a subpasta correspondente na pasta de destino
            pasta_categoria_destino = os.path.join(PASTA_PADRONIZADA, categoria)
            if not os.path.exists(pasta_categoria_destino):
                os.makedirs(pasta_categoria_destino)

            print(f"\nProcessando categoria: '{categoria}'")
            contador_categoria = 1

            # Itera sobre cada arquivo .txt na pasta da categoria
            for nome_arquivo in sorted(os.listdir(pasta_categoria_origem)): # sorted() para ordem consistente
                if nome_arquivo.endswith(".txt") and nome_arquivo.startswith(PREFIXO_NOME_ARQUIVO):
                    caminho_arquivo_original = os.path.join(pasta_categoria_origem, nome_arquivo)

                    # --- Extração e Substituição ---
                    try:
                        # Extrai o nome do interlocutor do nome do arquivo
                        nome_interlocutor = nome_arquivo.replace(PREFIXO_NOME_ARQUIVO, "").replace(".txt", "")

                        # Define o novo rótulo genérico (ex: "Amigo1")
                        novo_rotulo = f"{categoria}{contador_categoria}"

                        # Lê todo o conteúdo do arquivo original
                        with open(caminho_arquivo_original, 'r', encoding='utf-8') as f_origem:
                            conteudo = f_origem.read()

                        # Realiza as substituições usando regex para garantir que apenas o nome do autor seja substituído
                        # Substitui "Nome Interlocutor: " por "CategoriaN: "
                        conteudo_modificado = re.sub(
                            rf'({re.escape(nome_interlocutor)}):',
                            rf'{novo_rotulo}:',
                            conteudo
                        )
                        # Substitui "Seu Nome: " por "MeuNome: "
                        conteudo_modificado = re.sub(
                            rf'({re.escape(MEU_NOME)}):',
                            r'MeuNome:',
                            conteudo_modificado
                        )

                        # Define o nome e o caminho do novo arquivo padronizado
                        novo_nome_arquivo = f"{novo_rotulo}.txt"
                        caminho_arquivo_novo = os.path.join(pasta_categoria_destino, novo_nome_arquivo)

                        # Escreve o conteúdo modificado no novo arquivo
                        with open(caminho_arquivo_novo, 'w', encoding='utf-8') as f_destino:
                            f_destino.write(conteudo_modificado)

                        print(f"  - '{nome_arquivo}' -> '{novo_nome_arquivo}' (Interlocutor: '{nome_interlocutor}')")
                        contador_categoria += 1

                    except Exception as e:
                        print(f"  - [ERRO] Falha ao processar o arquivo '{nome_arquivo}': {e}")
                elif nome_arquivo.endswith(".txt"):
                    print(f"  - [AVISO] Arquivo '{nome_arquivo}' ignorado por não seguir o padrão de nome '{PREFIXO_NOME_ARQUIVO}...'.")


    print("-" * 50)
    print("Processo concluído com sucesso!")
    print(f"Verifique a pasta '{PASTA_PADRONIZADA}' para os arquivos anonimizados.")

# --- Bloco de Execução ---
if __name__ == "__main__":
    padronizar_conversas()