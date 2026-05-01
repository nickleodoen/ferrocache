.PHONY: build test cluster-up cluster-down cluster-test simulate simulate-no-ml simulate-cluster clean

build:
	cargo build --release

test:
	cargo test

cluster-up:
	docker compose up -d --build

cluster-down:
	docker compose down -v

cluster-test: cluster-up
	@echo "Waiting 5s for containers to start..."
	@sleep 5
	./tests/cluster_integration.sh

simulate:
	@echo "Installing dependencies..."
	pip install -q -r tests/requirements.txt
	@echo "Running simulation against localhost:3000..."
	python3 tests/simulate.py

simulate-no-ml:
	@echo "Running latency-only simulation (random vectors)..."
	python3 tests/simulate_no_ml.py

simulate-cluster:
	@echo "Running simulation against 3-node cluster (localhost:3001)..."
	python3 tests/simulate.py --url http://localhost:3001

clean:
	cargo clean
	docker compose down -v --rmi local
