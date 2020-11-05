SED_PWD=$(shell (echo $(shell PWD) | sed 's_/_\\/_g'))
POD_DC_NAME=kubectl -n dev get pods | grep data-collector | awk '{print $$1}'
POD_DB_NAME=kubectl -n dev get pods | grep database | awk '{print $$1}'
POD_TD_NAME=kubectl -n dev get pods | grep trader | awk '{print $$1}'

_mkdirs:
	@mkdir -p $(shell PWD)/storage/database
	@mkdir -p $(shell PWD)/k8s/deployments

_build_ymls:
	@cat $(shell PWD)/k8s/deployments-template/database-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/deployments/database.yml
	@cat $(shell PWD)/k8s/deployments-template/data_collector-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/deployments/data_collector.yml
	@cat $(shell PWD)/k8s/deployments-template/trader-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/deployments/trader.yml

_apply: _mkdirs _build_ymls
	@kubectl apply -f $(shell PWD)/k8s/admin/namespace.yml
	@kubectl apply -f $(shell PWD)/k8s/deployments/database.yml
	@kubectl apply -f $(shell PWD)/k8s/deployments/data_collector.yml
	@kubectl apply -f $(shell PWD)/k8s/deployments/trader.yml
	@kubectl apply -f $(shell PWD)/k8s/service/service.yml

_reapply: _mkdirs _build_ymls
	@kubectl delete -f $(shell PWD)/k8s/admin/namespace.yml
	@$(MAKE) _apply

_build_container:
	docker build dockerfiles -t binance_trader_services:latest
	docker pull postgres:latest

_run_if_not_exists:
ifneq ($(shell minikube status | grep host | cut -d ' ' -f 2),Running)
	@minikube start --mount-string="$(shell PWD):$(shell PWD)" --mount && \
	eval $$(minikube docker-env) && \
	$(MAKE) _build_container
endif
	@echo

install_on_mac:
	brew install kubectl minikube; \
	brew cask install virtualbox virtualbox-extension-pack

run: _run_if_not_exists
	@$(MAKE) _apply

stop:
	minikube stop

rm:
	minikube delete

pods:
	@kubectl -n dev get pods

db_bash:
	@kubectl -n dev exec -it $(shell $(POD_DB_NAME)) -- bash

dc_bash:
	@kubectl -n dev exec -it $(shell $(POD_DC_NAME)) -- bash

td_bash:
	@kubectl -n dev exec -it $(shell $(POD_TD_NAME)) -- bash