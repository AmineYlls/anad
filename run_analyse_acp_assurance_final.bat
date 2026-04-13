@echo off
cd /d %~dp0

echo Installation des dependances...
py -m pip install -r requirements_assurance_final.txt >nul 2>&1
if errorlevel 1 (
    python -m pip install -r requirements_assurance_final.txt
) else (
    py -m pip install -r requirements_assurance_final.txt
)

echo.
echo Lancement de l'analyse...
py analyse_acp_assurance_web_final.py DATA.xlsx --author "Remli" --project-title "Mini-projet ACP - Assurance automobile" >nul 2>&1
if errorlevel 1 (
    python analyse_acp_assurance_web_final.py DATA.xlsx --author "Remli" --project-title "Mini-projet ACP - Assurance automobile"
) else (
    py analyse_acp_assurance_web_final.py DATA.xlsx --author "Remli" --project-title "Mini-projet ACP - Assurance automobile"
)

echo.
echo Fichiers generes :
echo - rapport_acp_assurance_final.html
echo - resultats_acp_assurance_final.xlsx
pause
