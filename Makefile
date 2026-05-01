.PHONY: build test cluster-up cluster-down cluster-test clean

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

clean:
	cargo clean
	docker compose down -v --rmi local
