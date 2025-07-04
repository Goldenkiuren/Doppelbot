import re
import os
import json
import emoji
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

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO ---

def parsear_conversa_bruta(caminho_arquivo, meu_nome, outro_nome):
    mensagens_brutas = []
    padrao_regex = re.compile(
        r'^(\d{2}/\d{2}/\d{4},? \d{2}:\d{2}) - ([^:]+): (.*?)(?=(?:\r?\n)?^\d{2}/\d{2}/\d{4},? \d{2}:\d{2} - |\Z)',
        re.DOTALL | re.MULTILINE
    )
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
            stats_global["total_linhas_lidas"] += len(conteudo.split('\n'))
    except Exception as e:
        print(f"  [AVISO] Erro ao ler o arquivo {os.path.basename(caminho_arquivo)}: {e}")
        return []

    matches = padrao_regex.finditer(conteudo)
    for match in matches:
        timestamp_str, autor, texto_bruto = match.groups() 
        texto_bruto = texto_bruto.strip()
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
            texto_completo = "\n".join(bloco_atual["textos"])
            blocos.append({"autor": bloco_atual["autor"], "texto_completo_bruto": texto_completo, "timestamp": bloco_atual["timestamp_final"]})
            bloco_atual = {"autor": msg_atual["autor"], "textos": [msg_atual["texto_bruto"]], "timestamp_final": msg_atual["timestamp"]}
    
    texto_completo_final = "\n".join(bloco_atual["textos"]) # Alterado aqui
    blocos.append({"autor": bloco_atual["autor"], "texto_completo_bruto": texto_completo_final, "timestamp": bloco_atual["timestamp_final"]})
    stats_global["total_blocos_criados"] += len(blocos)
    return blocos

def filtrar_blocos_ai(blocos_brutos, meu_nome, outro_nome):
    blocos_filtrados = []
    # Usamos um iterador para poder avançar ele manualmente quando necessário
    iter_blocos = iter(enumerate(blocos_brutos)) 

    padrao_ai = re.compile(r'@(\d{10,})')

    for i, bloco_atual in iter_blocos:
        # Verifica se há um próximo bloco para evitar IndexError
        if i + 1 < len(blocos_brutos):
            bloco_seguinte = blocos_brutos[i+1]
            
            # Condição de descarte
            if (bloco_atual["autor"] == meu_nome and 
                padrao_ai.search(bloco_atual["texto_completo_bruto"]) and 
                bloco_seguinte["autor"] == outro_nome):
                
                stats_global["total_sequencias_ai_descartadas"] += 1
                # Pula o próximo bloco no iterador, efetivamente descartando ambos
                next(iter_blocos, None) 
                continue # Pula para a próxima iteração do for

        # Se não caiu na condição de descarte, adiciona o bloco atual
        blocos_filtrados.append(bloco_atual)
        
    return blocos_filtrados

def limpar_texto_e_validar(texto_bruto):
    # 1. Quebra o bloco em mensagens individuais usando o separador
    mensagens_individuais = texto_bruto.split("\n")
    
    mensagens_limpas = []
    
    # Lista de palavras-chave que invalidam uma mensagem inteira
    keywords_descarte = ["ligação de vídeo perdida", "mensagem apagada", "(arquivo anexado)", "ligação de voz perdida", "ligação de vídeo em grupo perdida"]
    
    # Lista de marcadores para remover, mantendo o resto do texto
    marcadores_remover = ["<Mídia oculta>", "<Mensagem editada>"]
    
    for msg in mensagens_individuais:
        msg_processada = msg.strip()
        
        # 2. Ignora mensagens que são exatamente "null" ou vazias
        if msg_processada.lower() == 'null' or not msg_processada:
            continue
            
        # 3. Descarta a mensagem se contiver keywords de descarte
        if any(keyword in msg_processada.lower() for keyword in keywords_descarte):
            continue
            
        # 4. Remove marcadores indesejados
        for marcador in marcadores_remover:
            msg_processada = msg_processada.replace(marcador, "")
            
        # 5. Remove links
        msg_processada = re.sub(r'https?://\S+', '', msg_processada)

        # 6. Remove Emojis
        msg_processada = emoji.replace_emoji(msg_processada, replace='')

        # Se após toda a limpeza a mensagem ainda tiver conteúdo, adicione-a
        if msg_processada.strip():
            mensagens_limpas.append(msg_processada.strip())

    # 6. Junta as mensagens limpas de volta com o separador
    return "\n".join(mensagens_limpas)


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
    if not os.path.isdir(PASTA_ENTRADA):
        print(f"ERRO: A pasta de entrada '{PASTA_ENTRADA}' não foi encontrada.")
        return

    dataset_completo = []
    print(f"Iniciando pré-processamento final a partir da pasta '{PASTA_ENTRADA}'...")
    print("-" * 50)

    for categoria in os.listdir(PASTA_ENTRADA):
        pasta_categoria = os.path.join(PASTA_ENTRADA, categoria)
        if os.path.isdir(pasta_categoria):
            print(f"\nProcessando categoria: '{categoria}'")
            for nome_arquivo in sorted(os.listdir(pasta_categoria)):
                if nome_arquivo.endswith(".txt"):
                    stats_global["total_arquivos_processados"] += 1
                    caminho_arquivo = os.path.join(pasta_categoria, nome_arquivo)
                    outro_nome = nome_arquivo.replace(".txt", "")
                    
                    print(f"  - Lendo arquivo: '{nome_arquivo}' (Interlocutor: {outro_nome})")

                    mensagens = parsear_conversa_bruta(caminho_arquivo, MEU_NOME_PADRONIZADO, outro_nome)
                    blocos = agrupar_mensagens(mensagens)
                    blocos_sem_ai = filtrar_blocos_ai(blocos, MEU_NOME_PADRONIZADO, outro_nome)
                    pares_processados = criar_e_validar_pares(blocos_sem_ai, MEU_NOME_PADRONIZADO)
                    
                    for par in pares_processados:
                        par['categoria'] = categoria
                        dataset_completo.append(par)

    stats_global["total_pares_finais"] = len(dataset_completo)

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


if __name__ == "__main__":
    processar_conversas_padronizadas()