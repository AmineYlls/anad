FICHIERS FINAUX - MINI-PROJET ACP

1) Installation
   python -m pip install -r requirements_assurance_final.txt

2) Exécution simple
   python analyse_acp_assurance_web_final.py DATA.xlsx --author "Votre nom" --project-title "Mini-projet ACP - Assurance automobile"

3) Fichiers générés
   - rapport_acp_assurance_final.html
   - resultats_acp_assurance_final.xlsx

4) Exécution Windows directe
   - double-cliquer sur run_analyse_acp_assurance_final.bat

5) Options utiles
   - choisir une feuille précise :
     python analyse_acp_assurance_web_final.py DATA.xlsx --sheet "car_insurance_claim.csv"

   - ne pas ouvrir automatiquement le rapport :
     python analyse_acp_assurance_web_final.py DATA.xlsx --no-open

   - exclure des colonnes supplémentaires :
     python analyse_acp_assurance_web_final.py DATA.xlsx --exclude COL1 COL2
