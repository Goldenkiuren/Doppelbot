import re
from datetime import datetime, timedelta

# --- CONSTANTES CONFIGURÁVEIS ---
MEU_NOME = "Augusto"
OUTRO_NOME = "Juliano" # Lembre-se de ajustar para cada conversa
THRESHOLD_RESPOSTA_HORAS = 5
MAX_LEN_INPUT = 2000
NOME_ARQUIVO_SAIDA = "dataset_processado.txt"
NOME_ARQUIVO_SAIDA_DESCARTADOS = "descartados_log.txt"

# --- DICIONÁRIO DE ESTATÍSTICAS (ATUALIZADO) ---
stats_counter = {
    "linhas_lidas_total": 0,
    "descartado_parse_falhou": 0,
    "descartado_sequencia_meta_ai": 0,
    "pares_potenciais_total": 0,
    "descartado_threshold_resposta": 0,
    "descartado_input_muito_longo": 0,
    "descartado_par_com_conteudo_invalido": 0
}

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO (VERSÃO 4.0) ---

def parsear_conversa_bruta(caminho_arquivo, nome_usuario1, nome_usuario2):
    """
    Fase 1: Apenas parseia o arquivo em uma estrutura de dados, mantendo o texto bruto.
    A única limpeza aqui é de linhas que não são mensagens válidas.
    """
    mensagens_brutas = []
    padrao_regex = re.compile(r'^((\d{2}/\d{2}/\d{4},? \d{2}:\d{2}) - ([^:]+): (.*))', re.MULTILINE)
    
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
            stats_counter["linhas_lidas_total"] = len(conteudo.split('\n'))
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return []

    matches = padrao_regex.finditer(conteudo)
    for match in matches:
        linha_completa, timestamp_str, autor, texto_bruto = match.groups()
        autor_limpo = autor.strip()

        if autor_limpo not in [nome_usuario1, nome_usuario2]:
            continue
        try:
            timestamp_str_sem_virgula = timestamp_str.replace(',', '')
            timestamp = datetime.strptime(timestamp_str_sem_virgula, '%d/%m/%Y %H:%M')
            mensagens_brutas.append({"timestamp": timestamp, "autor": autor_limpo, "texto_bruto": texto_bruto})
        except ValueError:
            stats_counter["descartado_parse_falhou"] += 1
            continue
    return mensagens_brutas

def agrupar_mensagens(mensagens_brutas):
    """Fase 2: Agrupa mensagens sequenciais do mesmo autor em 'blocos' de conversa."""
    if not mensagens_brutas: return []
    blocos = []
    bloco_atual = {"autor": mensagens_brutas[0]["autor"], "textos": [mensagens_brutas[0]["texto_bruto"]], "timestamp_final": mensagens_brutas[0]["timestamp"]}
    for i in range(1, len(mensagens_brutas)):
        msg_atual = mensagens_brutas[i]
        if msg_atual["autor"] == bloco_atual["autor"]:
            bloco_atual["textos"].append(msg_atual["texto_bruto"])
            bloco_atual["timestamp_final"] = msg_atual["timestamp"]
        else:
            blocos.append({"autor": bloco_atual["autor"], "texto_completo_bruto": "\n".join(bloco_atual["textos"]), "timestamp": bloco_atual["timestamp_final"]})
            bloco_atual = {"autor": msg_atual["autor"], "textos": [msg_atual["texto_bruto"]], "timestamp_final": msg_atual["timestamp"]}
    blocos.append({"autor": bloco_atual["autor"], "texto_completo_bruto": "\n".join(bloco_atual["textos"]), "timestamp": bloco_atual["timestamp_final"]})
    return blocos

def filtrar_blocos_ai(blocos_brutos, meu_nome, outro_nome, log_file):
    """
    Fase 3: Filtra sequências de interação com a Meta AI.
    Padrão: Bloco 'Meu Nome' com @, seguido de Bloco 'Outro Nome'.
    """
    blocos_filtrados = []
    i = 0
    padrao_ai = re.compile(r'@(\d{10,})') # Padrão para @ seguido de 10+ números
    
    while i < len(blocos_brutos):
        bloco_atual = blocos_brutos[i]
        # Verifica se há um próximo bloco para análise
        if i + 1 < len(blocos_brutos):
            bloco_seguinte = blocos_brutos[i+1]
            
            # Detecta o padrão: Eu pergunto para a IA, o outro "responde"
            if (bloco_atual["autor"] == meu_nome and 
                padrao_ai.search(bloco_atual["texto_completo_bruto"]) and 
                bloco_seguinte["autor"] == outro_nome):
                
                stats_counter["descartado_sequencia_meta_ai"] += 1
                log_descarte(log_file, "Sequência de Meta AI", f"PROMPT DESCARTADO:\n{bloco_atual['texto_completo_bruto']}\n\nRESPOSTA DESCARTADA:\n{bloco_seguinte['texto_completo_bruto']}")
                i += 2  # Pula os dois blocos (o seu e o da IA)
                continue
        
        blocos_filtrados.append(bloco_atual)
        i += 1
    return blocos_filtrados

def criar_e_validar_pares(blocos_filtrados, meu_nome, log_file):
    """
    Fase 4: Cria pares 'Outro -> Eu' e aplica toda a limpeza e validação de conteúdo.
    """
    pares_finais = []
    for i in range(1, len(blocos_filtrados)):
        bloco_anterior = blocos_filtrados[i-1]
        bloco_atual = blocos_filtrados[i]

        if bloco_anterior["autor"] != meu_nome and bloco_atual["autor"] == meu_nome:
            stats_counter["pares_potenciais_total"] += 1
            
            input_bruto = bloco_anterior["texto_completo_bruto"]
            output_bruto = bloco_atual["texto_completo_bruto"]
            
            # 1. Validação Temporal
            delta_tempo = bloco_atual["timestamp"] - bloco_anterior["timestamp"]
            if delta_tempo > timedelta(hours=THRESHOLD_RESPOSTA_HORAS):
                stats_counter["descartado_threshold_resposta"] += 1
                log_descarte(log_file, "Threshold de resposta excedido", f"INPUT:\n{input_bruto}\n\nOUTPUT:\n{output_bruto}")
                continue

            # 2. Limpeza do Conteúdo
            input_limpo = limpar_texto_e_validar(input_bruto)
            output_limpo = limpar_texto_e_validar(output_bruto)

            # 3. Validação do Conteúdo Limpo
            if not input_limpo or not output_limpo:
                stats_counter["descartado_par_com_conteudo_invalido"] += 1
                log_descarte(log_file, "Conteúdo inválido após limpeza (mídia, null, etc)", f"INPUT BRUTO:\n{input_bruto}\n\nOUTPUT BRUTO:\n{output_bruto}")
                continue
            
            if len(input_limpo) > MAX_LEN_INPUT:
                stats_counter["descartado_input_muito_longo"] += 1
                log_descarte(log_file, f"Input muito longo ({len(input_limpo)} chars)", f"INPUT LIMPO:\n{input_limpo}")
                continue
                
            pares_finais.append((input_limpo, output_limpo))
            
    return pares_finais

def limpar_texto_e_validar(texto_bruto):
    """Função auxiliar que limpa o texto e checa por padrões inválidos."""
    # Checa por mensagens de sistema que podem ter sido agrupadas
    if any(keyword in texto_bruto for keyword in ["Ligação de vídeo perdida", "Mensagem apagada", "(arquivo anexado)"]):
        return ""
    if ": null" in texto_bruto:
        return ""
        
    texto_limpo = texto_bruto.replace("<Mídia oculta>", "").strip()
    texto_limpo = re.sub(r'https?://\S+', '', texto_limpo).strip()
    return texto_limpo

def log_descarte(arquivo, motivo, conteudo):
    try:
        arquivo.write(f"--- MOTIVO: {motivo} ---\n")
        arquivo.write(f"CONTEÚDO:\n{conteudo}\n")
        arquivo.write("-" * 20 + "\n\n")
    except Exception: pass

# --- FUNÇÃO PRINCIPAL ---
def pre_processar_conversa_final(caminho_arquivo):
    """Orquestra o novo fluxo de pré-processamento."""
    print("Iniciando o pré-processamento (v4.0)...")
    
    with open(NOME_ARQUIVO_SAIDA_DESCARTADOS, 'w', encoding='utf-8') as log_file:
        log_file.write("--- Log de Descartados ---\n\n")
        
        # Fase 1: Parsear dados brutos
        mensagens_brutas = parsear_conversa_bruta(caminho_arquivo, MEU_NOME, OUTRO_NOME)
        print(f"1. {len(mensagens_brutas)} mensagens de autores válidos encontradas.")

        # Fase 2: Agrupar em blocos
        blocos_brutos = agrupar_mensagens(mensagens_brutas)
        print(f"2. {len(blocos_brutos)} blocos de conversa (turnos) criados.")
        
        # Fase 3: Filtrar interações com a IA
        blocos_filtrados = filtrar_blocos_ai(blocos_brutos, MEU_NOME, OUTRO_NOME, log_file)
        print(f"3. {len(blocos_filtrados)} blocos restantes após filtro da Meta AI.")

        # Fase 4: Criar e validar pares
        pares_finais = criar_e_validar_pares(blocos_filtrados, MEU_NOME, log_file)
        print(f"4. {len(pares_finais)} pares de input/output finais gerados.")

    # ... (print_stats_summary e salvamento do arquivo permanecem similares, precisam ser ajustados para os novos contadores) ...
    print("\nPré-processamento concluído!")
    return pares_finais

# --- Bloco de Execução ---
if __name__ == "__main__":
    CAMINHO_ARQUIVO_TXT = 'Exemplo2.txt'
    MEU_NOME = "Augusto"
    OUTRO_NOME = "Vitória"

    pares_processados = pre_processar_conversa_final(CAMINHO_ARQUIVO_TXT)
    
    if pares_processados:
        # Salvar o arquivo final...
        try:
            with open(NOME_ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
                for i, (entrada, saida) in enumerate(pares_processados):
                    f.write(f"----- PAR {i+1} -----\n")
                    f.write(f"INPUT:\n{entrada}\n\n")
                    f.write(f"OUTPUT:\n{saida}\n")
                    f.write("-" * 20 + "\n\n")
            print(f"\nSucesso! Os pares processados foram salvos em '{NOME_ARQUIVO_SAIDA}'.")
        except Exception as e:
            print(f"Ocorreu um erro ao salvar o arquivo: {e}")