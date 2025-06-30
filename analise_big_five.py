import os
import sys
import pandas as pd
import numpy as np

def analyze_personality_scores(folder_path, human_baseline_col, output_filename="analise_big_five_resultados.txt"):
    """
    Lê arquivos CSV, realiza uma análise comparativa avançada e genérica, e salva em um arquivo de texto.
    """
    if not os.path.isdir(folder_path):
        print(f"ERRO: O caminho '{folder_path}' não é um diretório válido.")
        sys.exit(1)

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"ERRO: Nenhum arquivo .csv encontrado no diretório '{folder_path}'.")
        sys.exit(1)

    print(f"Encontrados {len(csv_files)} arquivos CSV. Baseline humana definida como: '{human_baseline_col}'. Iniciando análise...")

    all_data = {}
    persona_columns = []

    # 1. Carregar todos os dados primeiro
    for filename in csv_files:
        try:
            # Limpa o nome do arquivo para obter a categoria, usando o nome da baseline fornecido
            category_name = os.path.splitext(filename)[0].replace(human_baseline_col, "").strip()
            df = pd.read_csv(os.path.join(folder_path, filename))
            
            if not persona_columns:
                persona_columns = [col for col in df.columns if col.lower() not in ['category', 'categoria', 'pergunta', 'texto']]
            
            all_data[category_name] = df[persona_columns]
        except Exception as e:
            print(f"AVISO: Falha ao carregar '{filename}': {e}")
            continue

    if not all_data:
        print("ERRO: Nenhum dado válido foi carregado. Saindo.")
        sys.exit(1)
        
    # Verifica se a coluna da baseline humana existe nos dados
    if human_baseline_col not in persona_columns:
        print(f"ERRO: A coluna da baseline humana '{human_baseline_col}' não foi encontrada nos arquivos CSV.")
        print(f"Colunas encontradas: {persona_columns}")
        sys.exit(1)

    # 2. Calcular as médias por categoria e criar uma tabela pivotada
    mean_scores_list = []
    for category, df in all_data.items():
        row = {'Traço de Personalidade': category}
        for persona in persona_columns:
            scores = pd.to_numeric(df[persona], errors='coerce').dropna()
            row[persona] = scores.mean()
        mean_scores_list.append(row)
    
    pivoted_means_df = pd.DataFrame(mean_scores_list).set_index('Traço de Personalidade').T
    pivoted_means_df.index.name = "Persona"
    pivoted_means_df = pivoted_means_df.round(2)

    # 3. Calcular as "distâncias de personalidade" em relação à baseline humana
    distance_results = {}
    human_vector = pivoted_means_df.loc[human_baseline_col]
    
    for persona in persona_columns:
        if persona != human_baseline_col:
            persona_vector = pivoted_means_df.loc[persona]
            abs_diff = (human_vector - persona_vector).abs().round(2)
            euclidean_dist = np.linalg.norm(human_vector - persona_vector).round(2)
            distance_results[persona] = {'diff_por_traco': abs_diff, 'distancia_total': euclidean_dist}

    # 4. Escrever tudo para o arquivo de saída
    print(f"Salvando análise em '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("====================================================\n")
        f.write("   ANÁLISE COMPARATIVA DE PERSONALIDADE (BIG FIVE)  \n")
        f.write("====================================================\n\n")

        f.write(f"Baseline Humana para Comparação: '{human_baseline_col}'\n\n")

        f.write("--- Tabela de Pontuações Médias ---\n")
        f.write("A tabela abaixo mostra a pontuação média para cada persona em cada um dos cinco traços de personalidade.\n\n")
        f.write(pivoted_means_df.to_string())
        f.write("\n\n")

        f.write(f"--- Análise de Distância da Personalidade (vs. {human_baseline_col}) ---\n")
        f.write("Esta seção mede o quão 'distante' cada IA está da personalidade humana de referência.\nValores menores indicam maior semelhança.\n\n")
        
        for persona, data in distance_results.items():
            f.write(f"--- Comparação: {persona} ---\n")
            f.write("Diferença Absoluta por Traço (quão longe em cada traço):\n")
            f.write(data['diff_por_traco'].to_string())
            f.write("\n\n")
            f.write(f"DISTÂNCIA EUCLIDIANA TOTAL (dissimilaridade geral): {data['distancia_total']}\n")
            f.write("(Este é um único número que resume a diferença total de personalidade. Quanto menor, mais parecida a IA é do humano de referência.)\n\n")

    print(f"\nAnálise concluída com sucesso. Resultados salvos em '{output_filename}'.")


# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python3 analise_big_five.py <caminho_para_a_pasta_com_csvs> <nome_da_coluna_humana>")
        print("Exemplo: python3 analise_big_five.py ./big_five_scores Augusto")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    human_column = sys.argv[2]
    analyze_personality_scores(input_folder, human_column)