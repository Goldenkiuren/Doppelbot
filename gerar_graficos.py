import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def process_data_for_plotting(folder_path, human_baseline_col):
    """Lê e processa os dados dos CSVs, retornando DataFrames prontos para plotagem."""
    if not os.path.isdir(folder_path):
        return None, None

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        return None, None

    all_data = {}
    persona_columns = []
    
    for filename in csv_files:
        try:
            category_name = os.path.splitext(filename)[0].replace(human_baseline_col, "").strip()
            df = pd.read_csv(os.path.join(folder_path, filename))
            
            if not persona_columns:
                persona_columns = [col for col in df.columns if col.lower() not in ['category', 'categoria', 'pergunta', 'texto']]
            
            all_data[category_name] = df[persona_columns]
        except Exception:
            continue
            
    if human_baseline_col not in persona_columns:
        print(f"ERRO: Coluna baseline '{human_baseline_col}' não encontrada.")
        sys.exit(1)

    mean_scores_list = []
    for category, df in all_data.items():
        row = {'Traço': category}
        for persona in persona_columns:
            scores = pd.to_numeric(df[persona], errors='coerce').dropna()
            row[persona] = scores.mean()
        mean_scores_list.append(row)
    
    pivoted_means_df = pd.DataFrame(mean_scores_list).set_index('Traço').T
    pivoted_means_df.index.name = "Persona"
    
    distance_results = {}
    human_vector = pivoted_means_df.loc[human_baseline_col]
    
    for persona in persona_columns:
        if persona != human_baseline_col:
            persona_vector = pivoted_means_df.loc[persona]
            abs_diff = (human_vector - persona_vector).abs()
            euclidean_dist = np.linalg.norm(human_vector - persona_vector)
            distance_results[persona] = {'diff_por_traco': abs_diff, 'distancia_total': euclidean_dist}

    return pivoted_means_df, distance_results

def plot_trait_comparison(df, filepath):
    """Gera um gráfico de barras agrupado comparando as pontuações médias por traço."""
    print(f"Gerando gráfico: {filepath}...")
    
    df_melted = df.reset_index().melt(id_vars='Persona', var_name='Traço', value_name='Pontuação Média')
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_melted, x='Traço', y='Pontuação Média', hue='Persona', palette='viridis')
    
    plt.title('Comparação de Pontuações Médias por Traço de Personalidade', fontsize=16)
    plt.ylabel('Pontuação Média', fontsize=12)
    plt.xlabel('Traço de Personalidade (Big Five)', fontsize=12)
    plt.xticks(rotation=10)
    plt.legend(title='Persona')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_difference(diff_series, persona_name, human_name, filepath, y_limit=None):
    """Gera um gráfico de barras mostrando a diferença de pontuação para uma IA."""
    print(f"Gerando gráfico: {filepath}...")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=diff_series.index, y=diff_series.values, palette='coolwarm_r')
    
    if y_limit is not None:
        plt.ylim(0, y_limit)
    
    plt.title(f'Diferença Absoluta de Pontuação: {human_name} vs. {persona_name}', fontsize=16)
    plt.ylabel('Diferença Absoluta na Média', fontsize=12)
    plt.xlabel('Traço de Personalidade', fontsize=12)
    plt.xticks(rotation=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_euclidean_distance(distances, filepath):
    """Gera um gráfico de barras comparando a distância Euclidiana total de cada IA."""
    print(f"Gerando gráfico: {filepath}...")
    
    personas = list(distances.keys())
    dist_values = [d['distancia_total'] for d in distances.values()]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=personas, y=dist_values, palette='magma')
    
    plt.title('Dissimilaridade de Personalidade Geral (Distância Euclidiana)', fontsize=16)
    plt.ylabel('Distância Total (Menor é Mais Similar)', fontsize=12)
    plt.xlabel('Modelo de IA', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def main(folder_path, human_baseline_col, output_dir="graficos"):
    """Função principal para orquestrar a análise e a geração de gráficos."""
    
    # --- ALTERAÇÃO PRINCIPAL AQUI ---
    # 1. Criar o diretório de saída se ele não existir
    if not os.path.exists(output_dir):
        print(f"Criando diretório de saída: '{output_dir}'")
        os.makedirs(output_dir)
        
    print("Iniciando processamento de dados para os gráficos...")
    pivoted_means, distance_results = process_data_for_plotting(folder_path, human_baseline_col)
    
    if pivoted_means is None:
        print("Não foi possível gerar os gráficos pois nenhum dado foi carregado.")
        return
        
    # 2. Determinar o limite máximo do eixo Y para os gráficos de diferença
    max_diff = 0
    for persona in distance_results:
        max_diff = max(max_diff, distance_results[persona]['diff_por_traco'].max())
    y_limit_for_diff_plots = max_diff * 1.1 

    # 3. Gerar gráficos salvando na pasta especificada
    plot_trait_comparison(pivoted_means, filepath=os.path.join(output_dir, "1_comparacao_por_traco.png"))
    
    if 'Doppelbot' in distance_results:
        plot_difference(distance_results['Doppelbot']['diff_por_traco'], 'Doppelbot', human_baseline_col,
                        filepath=os.path.join(output_dir, "2_diferenca_vs_doppelbot.png"),
                        y_limit=y_limit_for_diff_plots)
        
    if 'Llama' in distance_results:
        plot_difference(distance_results['Llama']['diff_por_traco'], 'Llama', human_baseline_col,
                        filepath=os.path.join(output_dir, "3_diferenca_vs_llama.png"),
                        y_limit=y_limit_for_diff_plots)
        
    plot_euclidean_distance(distance_results, filepath=os.path.join(output_dir, "4_comparacao_distancia_total.png"))
    
    print(f"\nTodos os gráficos foram gerados com sucesso na pasta '{output_dir}'.")

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python3 gerar_graficos.py <caminho_para_a_pasta_com_csvs> <nome_da_coluna_humana>")
        print("Exemplo: python3 gerar_graficos.py ./big_five_scores Augusto")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    human_column = sys.argv[2]
    main(input_folder, human_column)