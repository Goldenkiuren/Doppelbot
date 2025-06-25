# **Doppelbot: Pipeline de Pré-processamento para Fine-tuning de LLM**

Este projeto contém scripts para pré-processar logs de conversas do WhatsApp, transformando-os em um dataset estruturado para fine-tuning de modelos de linguagem.

O processo automatizado realiza a anonimização, limpeza, categorização e formatação dos dados.

### **Estrutura de Pastas Necessária**

/SeuProjeto/  
|  
|-- 📂 conversas\_originais/  
|   |-- 📂 Categoria1/  
|   |   |-- Conversa do WhatsApp com \[Nome da Pessoa\].txt  
|   |  
|   |-- 📂 Categoria2/  
|   |   |-- Conversa do WhatsApp com \[Outra Pessoa\].txt  
|  
|-- 📝 name\_normalize.py  
|-- 📝 pre\_processing.py  
|-- 📜 run\_pipeline.bat      (Para Windows)  
|-- 📜 run\_pipeline.sh      (Para Linux/macOS)  
|-- 📄 README.md

### **Como Executar**

1. **Organize os Arquivos:** Crie a pasta conversas\_originais e, dentro dela, subpastas para cada categoria (ex: Amigo, Trabalho). Mova os arquivos .txt para as pastas correspondentes.  
2. **Rode o Pipeline:** Abra um terminal na pasta raiz do projeto e use o comando para o seu sistema.

#### **Windows**

\# Formato: run\_pipeline.bat "Seu Nome" "pasta\_de\_origem"  
run\_pipeline.bat "NomePrincipal" "conversas\_originais"

#### **Linux / macOS / Git Bash**

\# Dar permissão de execução (apenas uma vez)  
chmod \+x run\_pipeline.sh

\# Formato: ./run\_pipeline.sh "Seu Nome" "pasta\_de\_origem"  
./run\_pipeline.sh "NomePrincipal" "conversas\_originais"

### **Saída**

O pipeline gera o arquivo dataset\_final.jsonl, contendo os dados limpos e prontos para o fine-tuning. Cada linha é um objeto JSON com as chaves input, output e categoria.