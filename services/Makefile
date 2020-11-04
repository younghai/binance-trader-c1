DB_CONTAINER_NAME=docker ps | grep postgres:latest | cut -d ' ' -f 1
DB_CONTAINER_NAMES=docker ps -a | grep postgres:latest | cut -d ' ' -f 1
SED_PWD=$(shell (echo $(shell PWD) | sed 's_/_\\/_g'))

POD_DC_NAME=kubectl -n dev get pods | grep data-collector | awk '{print $$1}'
POD_DB_NAME=kubectl -n dev get pods | grep database | awk '{print $$1}'
POD_TD_NAME=kubectl -n dev get pods | grep trader | awk '{print $$1}'

_build_container:
	docker build dockerfiles -t yanagi7393/binance_trader_services:latest

push_container: _build_container
	docker push yanagi7393/binance_trader_services:latest

_mkdirs:
	@mkdir -p $(shell PWD)/storage/database

_build_ymls:
	@cat $(shell PWD)/k8s/deployments-template/database-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/deployments/database.yml
	@cat $(shell PWD)/k8s/deployments-template/data_collector-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/deployments/data_collector.yml
	@cat $(shell PWD)/k8s/deployments-template/trader-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/deployments/trader.yml

_run_if_not_exists:
ifeq ($(shell minikube status | grep host | cut -d ' ' -f 2),)
	minikube start
endif
	@echo

_apply: _mkdirs _build_ymls
	@kubectl apply -f $(shell PWD)/k8s/admin/namespace.yml
	@kubectl apply -f $(shell PWD)/k8s/deployments/database.yml
	@kubectl apply -f $(shell PWD)/k8s/deployments/data_collector.yml
	@kubectl apply -f $(shell PWD)/k8s/deployments/trader.yml
	@kubectl apply -f $(shell PWD)/k8s/service/service.yml

install_on_mac:
	brew install kubectl minikube; \
	brew cask install virtualbox virtualbox-extension-pack

run: _run_if_not_exists
	@$(MAKE) _apply

rm:
	minikube delete

db_bash:
	@kubectl -n dev exec -it $(shell $(POD_DB_NAME)) -- bash

dc_bash:
	@kubectl -n dev exec -it $(shell $(POD_DC_NAME)) -- bash

td_bash:
	@kubectl -n dev exec -it $(shell $(POD_TD_NAME)) -- bash