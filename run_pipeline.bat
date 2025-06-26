@echo off
setlocal

:: Este script automatiza o pipeline completo: dados, formatação e fine-tuning.
:: Etapa 1: Normaliza e anonimiza os nomes.
:: Etapa 2: Gera o dataset base (input/output).
:: Etapa 3: Injeta a persona/instrução no dataset.
:: Etapa 4: Executa o fine-tuning com QLoRA.

:: --- VALIDAÇÃO DE ENTRADA ---
if "%~3"=="" (
    echo ERRO: Numero incorreto de parametros.
    echo.
    echo Uso: run_pipeline.bat "Seu Nome" "pasta_de_origem" "arquivo_de_descricao"
    echo Exemplo: run_pipeline.bat "Augusto" "conversas_originais" "descricao.txt"
    exit /b 1
)

set "USER_NAME=%~1"
set "SOURCE_FOLDER=%~2"
set "DESCRIPTION_FILE=%~3"

if not exist "%SOURCE_FOLDER%\" (
    echo ERRO: A pasta de origem '%SOURCE_FOLDER%' nao foi encontrada.
    exit /b 1
)
if not exist "%DESCRIPTION_FILE%" (
    echo ERRO: O arquivo de descricao '%DESCRIPTION_FILE%' nao foi encontrado.
    exit /b 1
)

:: --- EXECUÇÃO DO PIPELINE ---
echo =================================================
echo   PIPELINE COMPLETO DOPPELBOT
echo =================================================
echo Usuario a ser imitado: %USER_NAME%
echo Pasta de origem: %SOURCE_FOLDER%
echo Arquivo de Persona: %DESCRIPTION_FILE%
echo -------------------------------------------------

:: Etapa 1: Normalização
echo.
echo ^>^>^> ETAPA 1: Normalizando nomes e anonimizando...
python name_normalize.py "%USER_NAME%" "%SOURCE_FOLDER%"
if errorlevel 1 (echo ERRO FATAL na Etapa 1. & exit /b 1)

:: Etapa 2: Pré-processamento
echo.
echo ^>^>^> ETAPA 2: Gerando dataset base 'dataset_final.jsonl'...
python pre_processing.py
if errorlevel 1 (echo ERRO FATAL na Etapa 2. & exit /b 1)

:: Etapa 3: Injeção da Instrução
echo.
echo ^>^>^> ETAPA 3: Injetando persona em 'dataset_instruct.jsonl'...
python add_instruction.py "%DESCRIPTION_FILE%"
if errorlevel 1 (echo ERRO FATAL na Etapa 3. & exit /b 1)

:: Etapa 4: Fine-tuning
echo.
echo ^>^>^> ETAPA 4: INICIANDO FINE-TUNING DO MODELO. ESTA ETAPA PODE LEVAR HORAS.
python fine_tuning.py
if errorlevel 1 (echo ERRO FATAL na Etapa 4. & exit /b 1)

echo.
echo -------------------------------------------------
echo PIPELINE CONCLUIDO COM SUCESSO!
echo Os adaptadores do seu Doppelbot foram salvos na pasta 'doppelbot-llama3-8b-instruct-adapters'.
echo =================================================

endlocal