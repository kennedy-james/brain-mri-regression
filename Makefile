DB ?= optuna_study.db

setup:
	uv sync

start-jupyterlab:
	uv run jupyter lab

export-project:
	uv export --format requirements.txt > requirements.txt

start-optuna-dashboard:
	uv run optuna-dashboard sqlite:///$(DB)