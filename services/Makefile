DB_CONTAINER_NAME=docker ps | grep postgres:latest | cut -d ' ' -f 1
DB_CONTAINER_NAMES=docker ps -a | grep postgres:latest | cut -d ' ' -f 1
SED_PWD=$(shell (echo $(PWD) | sed 's_/_\\/_g'))


install_on_mac:
	brew install kubectl minikube; \
	brew cask install virtualbox virtualbox-extension-pack

_build_yamls:
	@cat $(shell PWD)/k8s/deployments/database-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/deployments/database.yml
	@cat $(shell PWD)/k8s/service/service-template.yml | sed 's/{{PWD}}/$(SED_PWD)/g' > $(shell PWD)/k8s/service/service.yml

_mkdirs:
	@mkdir -p $(shell PWD)/storage/database

_kube_start_if_not_exists:
ifeq ($(shell minikube status | grep host | cut -d ' ' -f 2),)
	minikube start
endif
	@echo

_kube_apply: _build_yamls _mkdirs
	@kubectl apply -f $(shell PWD)/k8s/admin/namespace.yml
	@kubectl apply -f $(shell PWD)/k8s/deployments/database.yml
	@kubectl apply -f $(shell PWD)/k8s/service/service.yml

kube_start: _kube_start_if_not_exists
	@$(MAKE) _kube_apply

kube_rm:
	minikube delete
