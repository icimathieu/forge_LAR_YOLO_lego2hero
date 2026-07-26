#!/bin/zsh
# Babysitter du run week-end — boucle de relance AUTOMATIQUE, détachée de Claude
# (même mécanisme que le run 25k du 21/06). Toutes les 60 s :
#   - si le log contient "WEEK-END TERMINÉ" → le babysitter s'arrête ;
#   - si weekend_full_matrix.sh ne tourne pas → relance (resume-total, log en append).
# Lancement (détaché, survit à la fermeture de Claude — seul l'arrêt machine le tue) :
#   nohup caffeinate -i zsh scripts/tools/weekend_babysitter.sh >/dev/null 2>&1 & disown
# Arrêt manuel : pkill -f weekend_babysitter
cd "$(dirname "$0")/../.."
LOG=output/weekend.log
while true; do
  if grep -q "WEEK-END TERMINÉ" $LOG 2>/dev/null; then
    echo "[babysitter $(date '+%d/%m %H:%M')] run terminé — arrêt du babysitter" >> $LOG
    exit 0
  fi
  if ! pgrep -f "zsh scripts/tools/weekend_full_matrix.sh" >/dev/null; then
    echo "[babysitter $(date '+%d/%m %H:%M')] run absent → RELANCE (resume)" >> $LOG
    caffeinate -i zsh scripts/tools/weekend_full_matrix.sh >> $LOG 2>&1 &
  fi
  sleep 60
done
