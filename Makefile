setup:
	uv sync

start-notebook:
	uv run jupyter notebook --port=8888

export-project:
	uv export --format requirements.txt > requirements.txt
