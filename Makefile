.PHONY: help install install-all test build-index run-webapp run-research run-echo check clean

help:
	@echo "Vicinal Development Commands:"
	@echo "  make install       - Install SDK (core only)"
	@echo "  make install-all   - Install SDK with all optional features (webapp, research, test)"
	@echo "  make build-index   - Compile the FAISS threat index from JSONL patterns"
	@echo "  make run-webapp    - Start the demo webapp (backend + frontend)"
	@echo "  make run-echo      - Start the demo webapp without an LLM (echo mode)"
	@echo "  make run-research  - Run the baseline comparison experiment"
	@echo "  make test          - Run the test suite"
	@echo "  make check         - Run ruff and mypy"
	@echo "  make clean         - Remove build artifacts and caches"

install:
	pip install -e .

install-all:
	pip install -e .[full,webapp,research,dev]
	cd webapp/frontend && npm install

build-index:
	python data/build_index.py

# Start FastAPI backend in background, Vite frontend in foreground
# Add VICINAL_MODEL_BACKEND=echo to bypass Ollama if you don't have it installed
run-webapp: build-index
	@echo "Starting FastAPI backend..."
	uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 & \
	echo "Starting Vite frontend..." && \
	cd webapp/frontend && npm run dev

run-echo: build-index
	@echo "Starting FastAPI backend (Echo Mode) ..."
	VICINAL_MODEL_BACKEND=echo uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 & \
	echo "Starting Vite frontend..." && \
	cd webapp/frontend && npm run dev

run-research: build-index
	python research/experiments/run_experiment.py

test:
	pytest tests/ -v

check:
	ruff check .
	mypy sdk/python/vicinal core engine webapp/backend

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/faiss_index.bin data/metadata.json
