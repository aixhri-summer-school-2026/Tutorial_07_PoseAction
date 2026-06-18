.PHONY: build run shell notebook logs down

# Automatically grab your host user's UID and GID
export UID := $(shell id -u)
export GID := $(shell id -g)

build:
	docker compose build

run:
	docker compose up -d

shell:
	docker compose exec reachy-mini-tutorial-07 /bin/bash

down:
	docker compose down

install-rules:
	bash scripts/camera_rules.sh && bash scripts/usb_permissions.sh
logs:
	docker compose logs -f reachy-mini-tutorial-07
	
# notebook:
#   docker compose exec reachy-mini jupyter notebook --ip=