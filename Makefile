# Makefile
.PHONY: help build start stop clean test train

help:
	@echo "Available commands:"
	@echo "  build     - Build all Docker containers"
	@echo "  start     - Start all services"
	@echo "  stop      - Stop all services"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  test      - Run tests"
	@echo "  train     - Train the classifier model"
	@echo "  logs      - Show logs from all services"

build:
	docker-compose build

start:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

test:
	docker-compose exec backend python -m pytest tests/

train:
	docker-compose exec backend python scripts/prepare_data.py
	docker-compose exec backend python scripts/train_classifier.py

logs:
	docker-compose logs -f

# Development setup
dev-setup:
	python -m venv venv
	source venv/bin/activate && pip install -r backend/requirements.txt
	cd frontend && npm install

dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm start