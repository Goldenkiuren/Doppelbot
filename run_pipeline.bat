@echo off
setlocal

:: Este script para Windows automatiza o pipeline completo de pré-processamento de dados.
:: Etapa 1: Roda o script de normalização para anonimizar as conversas.
:: Etapa 2: Roda o script de pré-processamento para gerar o dataset final.

:: --- VALIDAÇÃO DE ENTRADA ---
if "%~2"=="" (
    echo ERRO: Numero incorreto de parametros.
    echo.
    echo Uso: run_pipeline.bat "Seu Nome no WhatsApp" "caminho\para\pasta\originais"
    echo Exemplo: run_pipeline.bat "Augusto" "conversas_originais"
    exit /b 1
)

set "USER_NAME=%~1"
set "SOURCE_FOLDER=%~2"

if not exist "%SOURCE_FOLDER%\" (
    echo ERRO: A pasta de origem '%SOURCE_FOLDER%' nao foi encontrada.
    exit /b 1
)

:: --- EXECUÇÃO DO PIPELINE ---

echo =================================================
echo   PIPELINE DE PRE-PROCESSAMENTO DOPPELBOT
echo =================================================
echo Usuario a ser imitado: %USER_NAME%
echo Pasta de origem: %SOURCE_FOLDER%
echo -------------------------------------------------

:: Etapa 1: Normalização e Anonimização
echo.
echo ^>^>^> INICIANDO ETAPA 1: Normalizando os nomes dos arquivos...
python name_normalize.py "%USER_NAME%" "%SOURCE_FOLDER%"

:: Verifica se a Etapa 1 foi bem-sucedida antes de continuar
:: A verificação é simples: a pasta de saída deve existir.
if not exist "conversas_padronizadas\" (
    echo.
    echo ERRO FATAL: A pasta 'conversas_padronizadas' nao foi criada. A Etapa 1 falhou.
    exit /b 1
)

:: Etapa 2: Pré-processamento e Geração do Dataset Final
echo.
echo ^>^>^> INICIANDO ETAPA 2: Gerando o dataset final unificado...
python pre_processing.py

echo.
echo -------------------------------------------------
echo PIPELINE CONCLUIDO COM SUCESSO!
echo O dataset final foi salvo em 'dataset_final.jsonl'.
echo =================================================

endlocal
