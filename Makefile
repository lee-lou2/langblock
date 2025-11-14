.PHONY: help install format lint test clean ingestion search demo

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies using uv"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Run linting checks"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove cache and build files"
	@echo "  make ingestion  - Run Q&A example ingestion"
	@echo "  make search     - Run Q&A example search"
	@echo "  make demo       - Run full demo (ingestion + search)"

install:
	uv sync

format:
	uv run black .

lint:
	uv run black --check .

test:
	uv run pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .lancedb

ingestion:
	uv run python examples/qna/ingestion.py

search:
	uv run python examples/qna/main.py

demo: ingestion search