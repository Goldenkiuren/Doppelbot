import re
import os
import json
from datetime import datetime, timedelta

#CONSTANTES DE CONFIGURAÇÃO
#Nome (padronizado pelo normalize)
MEU_NOME_PADRONIZADO = "MeuNome"

# Nome da pasta que contém os arquivos já padronizados e anonimizados.
PASTA_ENTRADA = "conversas_padronizadas"

# Nome do arquivo final que conterá o dataset completo para o fine-tuning.
ARQUIVO_SAIDA_JSONL = "dataset_final.jsonl"

# Hiperparâmetros de filtragem (validados anteriormente).
THRESHOLD_RESPOSTA_HORAS = 5
MAX_LEN_INPUT = 2000

# --- DICIONÁRIO GLOBAL DE ESTATÍSTICAS ---
# Será usado para acumular as estatísticas de todos os arquivos processados.
stats_global = {
    "total_arquivos_processados": 0,
    "total_linhas_lidas": 0,
    "total_blocos_criados": 0,
    "total_sequencias_ai_descartadas": 0,
    "total_pares_potenciais": 0,
    "total_pares_descartados_tempo": 0,
    "total_pares_descartados_tamanho": 0,
    "total_pares_descartados_conteudo": 0,
    "total_pares_finais": 0
}

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO (REUTILIZADAS DA v4.0) ---

def parsear_conversa_bruta(caminho_arquivo, meu_nome, outro_nome):
    mensagens_brutas = []
    padrao_regex = re.compile(r'^((\d{2}/\d{2}/\d{4},? \d{2}:\d{2}) - ([^:]+): (.*))', re.MULTILINE)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
            stats_global["total_linhas_lidas"] += len(conteudo.split('\n'))
    except Exception as e:
        print(f"  [AVISO] Erro ao ler o arquivo {os.path.basename(caminho_arquivo)}: {e}")
        return []

    matches = padrao_regex.finditer(conteudo)
    for match in matches:
        _, timestamp_str, autor, texto_bruto = match.groups()
        autor_limpo = autor.strip()
        if autor_limpo not in [meu_nome, outro_nome]:
            continue
        try:
            timestamp = datetime.strptime(timestamp_str.replace(',', ''), '%d/%m/%Y %H:%M')
            mensagens_brutas.append({"timestamp": timestamp, "autor": autor_limpo, "texto_bruto": texto_bruto})
        except ValueError:
            continue
    return mensagens_brutas

def agrupar_mensagens(mensagens_brutas):
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
    stats_global["total_blocos_criados"] += len(blocos)
    return blocos

def filtrar_blocos_ai(blocos_brutos, meu_nome, outro_nome):
    blocos_filtrados = []
    i = 0
    padrao_ai = re.compile(r'@(\d{10,})')
    descartes_ai_neste_arquivo = 0
    while i < len(blocos_brutos):
        bloco_atual = blocos_brutos[i]
        if i + 1 < len(blocos_brutos):
            bloco_seguinte = blocos_brutos[i+1]
            if (bloco_atual["autor"] == meu_nome and padrao_ai.search(bloco_atual["texto_completo_bruto"]) and bloco_seguinte["autor"] == outro_nome):
                descartes_ai_neste_arquivo += 1
                i += 2
                continue
        blocos_filtrados.append(bloco_atual)
        i += 1
    stats_global["total_sequencias_ai_descartadas"] += descartes_ai_neste_arquivo
    return blocos_filtrados

def limpar_texto_e_validar(texto_bruto):
    if any(keyword in texto_bruto for keyword in ["Ligação de vídeo perdida", "Mensagem apagada", "(arquivo anexado)"]) or ": null" in texto_bruto:
        return ""
    texto_limpo = texto_bruto.replace("<Mídia oculta>", "").strip()
    texto_limpo = re.sub(r'https?://\S+', '', texto_limpo).strip()
    return texto_limpo

def criar_e_validar_pares(blocos_filtrados, meu_nome):
    pares_finais = []
    for i in range(1, len(blocos_filtrados)):
        bloco_anterior = blocos_filtrados[i-1]
        bloco_atual = blocos_filtrados[i]
        if bloco_anterior["autor"] != meu_nome and bloco_atual["autor"] == meu_nome:
            stats_global["total_pares_potenciais"] += 1
            delta_tempo = bloco_atual["timestamp"] - bloco_anterior["timestamp"]
            if delta_tempo > timedelta(hours=THRESHOLD_RESPOSTA_HORAS):
                stats_global["total_pares_descartados_tempo"] += 1
                continue

            input_limpo = limpar_texto_e_validar(bloco_anterior["texto_completo_bruto"])
            output_limpo = limpar_texto_e_validar(bloco_atual["texto_completo_bruto"])

            if not input_limpo or not output_limpo:
                stats_global["total_pares_descartados_conteudo"] += 1
                continue
            if len(input_limpo) > MAX_LEN_INPUT:
                stats_global["total_pares_descartados_tamanho"] += 1
                continue
            
            pares_finais.append({"input": input_limpo, "output": output_limpo})
    return pares_finais

# --- ORQUESTRADOR PRINCIPAL ---

def processar_conversas_padronizadas():
    """
    Orquestra o processo completo: itera sobre as pastas e arquivos padronizados,
    processa cada um e agrega os resultados em um único arquivo JSONL.
    """
    if not os.path.isdir(PASTA_ENTRADA):
        print(f"ERRO: A pasta de entrada '{PASTA_ENTRADA}' não foi encontrada.")
        print("Certifique-se de que a pasta com as conversas padronizadas existe.")
        return

    dataset_completo = []
    print(f"Iniciando pré-processamento final a partir da pasta '{PASTA_ENTRADA}'...")
    print("-" * 50)

    # Itera sobre as subpastas (categorias)
    for categoria in os.listdir(PASTA_ENTRADA):
        pasta_categoria = os.path.join(PASTA_ENTRADA, categoria)
        if os.path.isdir(pasta_categoria):
            print(f"\nProcessando categoria: '{categoria}'")
            
            # Itera sobre os arquivos .txt na categoria
            for nome_arquivo in sorted(os.listdir(pasta_categoria)):
                if nome_arquivo.endswith(".txt"):
                    stats_global["total_arquivos_processados"] += 1
                    caminho_arquivo = os.path.join(pasta_categoria, nome_arquivo)
                    # O nome do interlocutor é o nome do arquivo sem a extensão
                    outro_nome = nome_arquivo.replace(".txt", "")
                    
                    print(f"  - Lendo arquivo: '{nome_arquivo}' (Interlocutor: {outro_nome})")

                    # Pipeline de processamento para um arquivo
                    mensagens = parsear_conversa_bruta(caminho_arquivo, MEU_NOME_PADRONIZADO, outro_nome)
                    blocos = agrupar_mensagens(mensagens)
                    blocos_sem_ai = filtrar_blocos_ai(blocos, MEU_NOME_PADRONIZADO, outro_nome)
                    pares_processados = criar_e_validar_pares(blocos_sem_ai, MEU_NOME_PADRONIZADO)
                    
                    # Adiciona a categoria a cada par e agrega ao dataset final
                    for par in pares_processados:
                        par['categoria'] = categoria
                        dataset_completo.append(par)

    stats_global["total_pares_finais"] = len(dataset_completo)

    # Salva o dataset final em formato JSONL
    try:
        with open(ARQUIVO_SAIDA_JSONL, 'w', encoding='utf-8') as f:
            for par in dataset_completo:
                f.write(json.dumps(par, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"\n[ERRO FATAL] Ocorreu um erro ao salvar o arquivo final: {e}")
        return

    print("-" * 50)
    print("Processamento e unificação concluídos com sucesso!")
    print(f"\n--- ESTATÍSTICAS GLOBAIS ---")
    print(f"Arquivos processados: {stats_global['total_arquivos_processados']}")
    print(f"Pares de treino finais gerados: {stats_global['total_pares_finais']}")
    print(f"Pares potenciais encontrados: {stats_global['total_pares_potenciais']}")
    print(f"  - Descartados por tempo (> {THRESHOLD_RESPOSTA_HORAS}h): {stats_global['total_pares_descartados_tempo']}")
    print(f"  - Descartados por conteúdo inválido: {stats_global['total_pares_descartados_conteudo']}")
    print(f"  - Descartados por tamanho do input (> {MAX_LEN_INPUT} chars): {stats_global['total_pares_descartados_tamanho']}")
    print(f"  - Interações com Meta AI removidas: {stats_global['total_sequencias_ai_descartadas']}")
    print("-" * 50)
    print(f"Seu dataset final está pronto em '{ARQUIVO_SAIDA_JSONL}'.")

# --- BLOCO DE EXECUÇÃO ---
if __name__ == "__main__":
    processar_conversas_padronizadas()

