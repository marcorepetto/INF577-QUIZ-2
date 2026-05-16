#!/usr/bin/env bash
set -euo pipefail

# Ejecuta main.py
.venv/bin/python main.py && notify-send "Entrenamiento finalizado" && paplay --volume=36768 ~/Música/notificaciones/positiva_0.wav

# Preguntar si subir submission a Kaggle
read -r -p "¿Desea subir la submission a Kaggle? [y/N]: " answer
case "$answer" in
	[Yy]* )
		if command -v kaggle >/dev/null 2>&1; then
			kaggle competitions submit -c espejito-espejito-inf577 -f outputs/submission.csv -m ""
            python utils/kaggle.py && notify-send "Resultados obtenidos" && paplay --volume=36768 ~/Música/notificaciones/positiva_0.wav
		else
			echo "kaggle CLI no está instalada. Instale kaggle para poder subir submissions."
			exit 1
		fi
		;;
	* )
		echo "No se subió la submission."
		;;
esac



