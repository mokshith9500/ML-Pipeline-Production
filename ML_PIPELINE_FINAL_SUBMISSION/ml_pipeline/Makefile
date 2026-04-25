# =============================================================================
# Makefile — ML Pipeline
# Provides simple commands for common pipeline operations
# =============================================================================

.PHONY: help setup data run run-date clean-outputs lint

# Default target
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║        ML Pipeline — Available Commands              ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make setup        Install dependencies"
	@echo "  make data         Generate sample data in data/raw/"
	@echo "  make run          Run the full pipeline (all new files)"
	@echo "  make run-date D=YYYY-MM-DD   Run for a specific date"
	@echo "  make dry-run      Run without training (validation only)"
	@echo "  make clean        Remove generated outputs"
	@echo "  make lint         Check code style with flake8"
	@echo ""

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Setup complete."

data:
	@echo "Generating sample data..."
	python generate_sample_data.py

run:
	@echo "Starting full pipeline..."
	python main.py

run-date:
	@echo "Starting pipeline for date: $(D)"
	python main.py --date $(D)

dry-run:
	@echo "Starting dry run (no training)..."
	python main.py --dry-run

clean:
	@echo "Cleaning generated outputs..."
	rm -f data/processed/*.csv
	rm -f data/processed_files.json
	rm -f data/baseline_stats.json
	rm -f models/*.pkl
	rm -f models/metrics_registry.json
	rm -f logs/*.log
	rm -f logs/*.json
	@echo "✓ Clean complete."

lint:
	@echo "Running flake8..."
	flake8 src/ main.py --max-line-length=100 --ignore=E501,W503
