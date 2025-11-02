setup:
	uv sync

start-notebook:
	uv run jupyter notebook --port=8888

export-project:
	uv export --format requirements.txt > requirements.txt

start-optuna-dashboard:
	@if [ -z "$(DB)" ]; then \
		echo "Set DB env var to a valid optuna study database path. E.g. DB=optuna_study.db"; \
		exit 1; \
	fi
	uv run optuna-dashboard sqlite:///$(DB)