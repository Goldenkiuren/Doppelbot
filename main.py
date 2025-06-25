import re
from datetime import datetime, timedelta

# --- CONSTANTES CONFIGURÁVEIS ---
MEU_NOME = "Augusto"
OUTRO_NOME = "Juliano"
THRESHOLD_RESPOSTA_HORAS = 5
MAX_LEN_INPUT = 2000
NOME_ARQUIVO_SAIDA = "dataset_processado.txt"

MENSAGENS_SISTEMA_IGNORAR = [
    "As mensagens e ligações são protegidas com a criptografia de ponta a ponta.",
    "Ligação de vídeo perdida",
    "Ligação de voz perdida",
    "Mensagem apagada",
    "Você mudou para as mensagens temporárias.",
    "(arquivo anexado)",
]

# Dicionário para rastrear estatísticas de descarte
stats_counter = {
    "mensagens_parseadas_total": 0,
    "descartado_autor_invalido": 0,
    "descartado_texto_vazio_ou_midia_isolada": 0,
    "descartado_palavra_chave_sistema": 0,
    "descartado_null": 0,
    "pares_potenciais_total": 0,
    "descartado_threshold_resposta": 0,
    "descartado_input_muito_longo": 0
}


def limpar_texto_mensagem(texto):
    """Limpa o conteúdo de uma única mensagem de texto."""
    texto_limpo = texto.replace("<Mídia oculta>", "").strip()
    texto_limpo = re.sub(r'https?://\S+', '', texto_limpo).strip()
    return texto_limpo

def parsear_conversa(caminho_arquivo, nome_usuario1, nome_usuario2):
    """
    Lê um arquivo de log do WhatsApp e o transforma em uma lista estruturada,
    contando os descartes na fase inicial.
    """
    mensagens_estruturadas = []
    padrao_regex = re.compile(r'^(\d{2}/\d{2}/\d{4},? \d{2}:\d{2}) - ([^:]+): (.*)', re.MULTILINE)

    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return []

    # O regex irá lidar com as linhas, não precisamos pré-processar o conteúdo todo.
    matches = padrao_regex.finditer(conteudo)
    
    for match in matches:
        stats_counter["mensagens_parseadas_total"] += 1
        timestamp_str, autor, texto_bruto = match.groups()

        autor_limpo = autor.strip()
        if autor_limpo not in [nome_usuario1, nome_usuario2]:
            stats_counter["descartado_autor_invalido"] += 1
            continue

        if texto_bruto.strip() == "null":
            stats_counter["descartado_null"] += 1
            continue

        if any(keyword in texto_bruto for keyword in MENSAGENS_SISTEMA_IGNORAR):
            stats_counter["descartado_palavra_chave_sistema"] += 1
            continue

        texto_limpo = limpar_texto_mensagem(texto_bruto)
        if not texto_limpo:
            stats_counter["descartado_texto_vazio_ou_midia_isolada"] += 1
            continue

        try:
            timestamp_str_sem_virgula = timestamp_str.replace(',', '')
            timestamp = datetime.strptime(timestamp_str_sem_virgula, '%d/%m/%Y %H:%M')
            mensagens_estruturadas.append({
                "timestamp": timestamp,
                "autor": autor_limpo,
                "texto": texto_limpo
            })
        except ValueError:
            stats_counter["descartado_autor_invalido"] += 1 # Conta linhas mal formatadas como inválidas
            continue
            
    return mensagens_estruturadas

def agrupar_mensagens(mensagens_estruturadas):
    """
    Agrupa mensagens consecutivas do mesmo autor em blocos de conversa.
    """
    if not mensagens_estruturadas:
        return []
    blocos = []
    bloco_atual = {
        "autor": mensagens_estruturadas[0]["autor"],
        "textos": [mensagens_estruturadas[0]["texto"]],
        "timestamp_final": mensagens_estruturadas[0]["timestamp"]
    }
    for i in range(1, len(mensagens_estruturadas)):
        msg_atual = mensagens_estruturadas[i]
        if msg_atual["autor"] == bloco_atual["autor"]:
            bloco_atual["textos"].append(msg_atual["texto"])
            bloco_atual["timestamp_final"] = msg_atual["timestamp"]
        else:
            blocos.append({
                "autor": bloco_atual["autor"],
                "texto_completo": "\n".join(bloco_atual["textos"]),
                "timestamp": bloco_atual["timestamp_final"]
            })
            bloco_atual = {
                "autor": msg_atual["autor"],
                "textos": [msg_atual["texto"]],
                "timestamp_final": msg_atual["timestamp"]
            }
    blocos.append({
        "autor": bloco_atual["autor"],
        "texto_completo": "\n".join(bloco_atual["textos"]),
        "timestamp": bloco_atual["timestamp_final"]
    })
    return blocos

def criar_pares_input_output(blocos_agrupados, meu_nome):
    """
    Cria pares de (input, output) e rastreia os descartes nesta fase.
    """
    pares = []
    for i in range(1, len(blocos_agrupados)):
        bloco_anterior = blocos_agrupados[i-1]
        bloco_atual = blocos_agrupados[i]

        if bloco_anterior["autor"] != meu_nome and bloco_atual["autor"] == meu_nome:
            stats_counter["pares_potenciais_total"] += 1
            
            delta_tempo = bloco_atual["timestamp"] - bloco_anterior["timestamp"]
            if delta_tempo > timedelta(hours=THRESHOLD_RESPOSTA_HORAS):
                stats_counter["descartado_threshold_resposta"] += 1
                continue

            input_text = bloco_anterior["texto_completo"]
            output_text = bloco_atual["texto_completo"]
            
            if len(input_text) > MAX_LEN_INPUT:
                stats_counter["descartado_input_muito_longo"] += 1
                continue
            
            if not input_text or not output_text:
                # Esta verificação é redundante se o parsear_conversa funcionar bem, mas é uma boa segurança.
                continue

            pares.append((input_text, output_text))
    return pares

def print_stats_summary():
    """Imprime um sumário das estatísticas de pré-processamento."""
    print("\n--- Sumário do Pré-Processamento ---")
    print(f"Total de mensagens no arquivo original: {stats_counter['mensagens_parseadas_total']}")
    
    print("\n--- Fase 1: Limpeza Inicial de Mensagens ---")
    descartados_fase1 = (
        stats_counter["descartado_autor_invalido"] + 
        stats_counter["descartado_null"] + 
        stats_counter["descartado_palavra_chave_sistema"] + 
        stats_counter["descartado_texto_vazio_ou_midia_isolada"]
    )
    print(f"Mensagens descartadas na limpeza inicial: {descartados_fase1}")
    print(f"  - Por autor inválido ou linha mal formatada: {stats_counter['descartado_autor_invalido']}")
    print(f"  - Por conter 'null': {stats_counter['descartado_null']}")
    print(f"  - Por palavras-chave do sistema (chamadas, etc.): {stats_counter['descartado_palavra_chave_sistema']}")
    print(f"  - Por estarem vazias ou serem apenas mídia/link: {stats_counter['descartado_texto_vazio_ou_midia_isolada']}")
    
    print("\n--- Fase 2: Filtragem de Pares de Conversa ---")
    print(f"Total de pares 'Outro -> Eu' encontrados: {stats_counter['pares_potenciais_total']}")
    descartados_fase2 = (
        stats_counter["descartado_threshold_resposta"] + 
        stats_counter["descartado_input_muito_longo"]
    )
    print(f"Pares descartados na filtragem: {descartados_fase2}")
    print(f"  - Por resposta exceder o threshold de {THRESHOLD_RESPOSTA_HORAS} horas: {stats_counter['descartado_threshold_resposta']}")
    print(f"  - Por input exceder o limite de {MAX_LEN_INPUT} caracteres: {stats_counter['descartado_input_muito_longo']}")
    print("-" * 35)


def pre_processar_conversa(caminho_arquivo):
    """
    Função principal que orquestra todo o processo de pré-processamento.
    """
    print("Iniciando o pré-processamento...")
    
    mensagens = parsear_conversa(caminho_arquivo, MEU_NOME, OUTRO_NOME)
    if not mensagens:
        print("Nenhuma mensagem válida encontrada. Encerrando.")
        return []
    print(f"1. {len(mensagens)} mensagens válidas parseadas (após limpeza inicial).")

    blocos = agrupar_mensagens(mensagens)
    print(f"2. {len(blocos)} blocos de conversa criados.")

    pares_finais = criar_pares_input_output(blocos, MEU_NOME)
    print(f"3. {len(pares_finais)} pares de input/output gerados.")
    
    print_stats_summary()
    
    print("Pré-processamento concluído!")
    return pares_finais


if __name__ == "__main__":
    CAMINHO_ARQUIVO_TXT = 'Exemplo de Conversa.txt'

    pares_processados = pre_processar_conversa(CAMINHO_ARQUIVO_TXT)
    
    if pares_processados:
        try:
            with open(NOME_ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
                f.write(f"--- Dataset Processado: {len(pares_processados)} pares ---\n")
                f.write(f"MEU_NOME: {MEU_NOME}\n")
                f.write(f"OUTRO_NOME: {OUTRO_NOME}\n")
                f.write("-" * 40 + "\n\n")

                for i, (entrada, saida) in enumerate(pares_processados):
                    f.write(f"----- PAR {i+1} -----\n")
                    f.write(f"INPUT ({OUTRO_NOME}):\n{entrada}\n\n")
                    f.write(f"OUTPUT ({MEU_NOME}):\n{saida}\n")
                    f.write("-" * 20 + "\n\n")
            print(f"\nSucesso! Os pares processados foram salvos em '{NOME_ARQUIVO_SAIDA}'.")
        except Exception as e:
            print(f"\nOcorreu um erro ao salvar o arquivo: {e}")