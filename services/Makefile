OS_NAME=$(shell (python -c "import platform; print(platform.system())"))
CPU_CORES=$(shell (python -c "import multiprocessing; print(multiprocessing.cpu_count())"))

PARENT_PWD=$(shell dirname $(shell pwd))
SED_PWD=$(shell (echo $(shell pwd) | sed 's_/_\\/_g'))
SED_PARENT_PWD=$(shell (dirname $(shell pwd) | sed 's_/_\\/_g'))

POD_DC_NAME=kubectl -n dev get pods | grep data-collector | awk '{print $$1}'
POD_DB_NAME=kubectl -n dev get pods | grep database | awk '{print $$1}'
POD_TD_NAME=kubectl -n dev get pods | grep trader | awk '{print $$1}'

define GENERATE_YML
	cat "$(shell pwd)/k8s/deployments-template/$1-template.yml" |\
	sed "s/{{PWD}}/$(SED_PWD)/g" |\
	sed "s/{{PARENT_PWD}}/$(SED_PARENT_PWD)/g" |\
	sed "s/{{EXP_NAME}}/$2/g" |\
	sed "s/{{REPORT_PREFIX}}/$3/g" |\
	sed "s/{{REPORT_BASE_CURRENCY}}/$4/g" |\
	sed "s/{{REPORT_ID}}/\"$5\"/g" >\
	$(shell pwd)/k8s/deployments/$1.yml
endef

define BUILD_SECRET
	mkdir -p /tmp/k8s_secret; \
	printf $1 > /tmp/k8s_secret/api_key; \
	printf $2 > /tmp/k8s_secret/secret_key; \
	printf $3 > /tmp/k8s_secret/test_mode; \
	kubectl -n dev create secret generic market-secret --from-file=/tmp/k8s_secret/api_key --from-file=/tmp/k8s_secret/secret_key --from-file=/tmp/k8s_secret/test_mode; \
	rm -rf /tmp/k8s_secret
endef

_mkdirs:
	@sudo chmod -R ug+rw $(PARENT_PWD)
	@mkdir -p $(shell pwd)/storage/database
	@mkdir -p $(shell pwd)/storage/trader
	@mkdir -p $(shell pwd)/k8s/deployments

_build_ymls:
	@read -p "EXP_NAME: " EXP_NAME; EXP_NAME=$${EXP_NAME:-v002}; \
	read -p "REPORT_PREFIX: " REPORT_PREFIX; REPORT_PREFIX=$${REPORT_PREFIX:-V1_CSET1}; \
	read -p "REPORT_BASE_CURRENCY: " REPORT_BASE_CURRENCY; REPORT_BASE_CURRENCY=$${REPORT_BASE_CURRENCY:-USDT}; \
	read -p "REPORT_ID: " REPORT_ID; REPORT_ID=$${REPORT_ID:-158}; \
	$(call GENERATE_YML,database,$$EXP_NAME,$$REPORT_PREFIX,$$REPORT_BASE_CURRENCY,$$REPORT_ID);\
	$(call GENERATE_YML,data_collector,$$EXP_NAME,$$REPORT_PREFIX,$$REPORT_BASE_CURRENCY,$$REPORT_ID);\
	$(call GENERATE_YML,trader,$$EXP_NAME,$$REPORT_PREFIX,$$REPORT_BASE_CURRENCY,$$REPORT_ID)

_apply_secret:
ifeq ($(shell kubectl -n dev get secret market-secret 2> /dev/null),)
	@read -p "EXCHANGE_API_KEY: " EXCHANGE_API_KEY; \
	read -p "EXCHANGE_SECRET_KEY: " EXCHANGE_SECRET_KEY; \
	read -p "TEST_MODE: " TEST_MODE; \
	$(call BUILD_SECRET,$$EXCHANGE_API_KEY,$$EXCHANGE_SECRET_KEY,$$TEST_MODE)
endif
	@echo

_apply:
	@kubectl apply -f $(shell pwd)/k8s/admin/namespace.yml
	@$(MAKE) _apply_secret

	@kubectl apply -f $(shell pwd)/k8s/deployments/database.yml
	@kubectl apply -f $(shell pwd)/k8s/deployments/data_collector.yml
	@kubectl apply -f $(shell pwd)/k8s/deployments/trader.yml
	@kubectl apply -f $(shell pwd)/k8s/service/service.yml

_build_container:
	docker build dockerfiles -t binance_trader_services:latest
	docker pull postgres:latest

_set_minikube_config:
	minikube config set memory 8192
	minikube config set cpus $(CPU_CORES)

_run_if_not_exists: _mkdirs _build_ymls _set_minikube_config
ifneq ($(shell minikube status | grep host | cut -d ' ' -f 2),Running)
	@minikube start --mount-string="$(PARENT_PWD):$(PARENT_PWD)" --mount && \
	eval $$(minikube docker-env) && \
	$(MAKE) _build_container
endif
	@echo

_install_on_mac:
	brew install kubectl minikube
	brew cask install virtualbox virtualbox-extension-pack

_install_on_ubuntu:
	sudo apt update -y
	sudo apt install -y virtualbox virtualbox-ext-pack

	wget https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
	sudo mv minikube-linux-amd64 /usr/local/bin/minikube
	sudo chmod +x /usr/local/bin/minikube

	curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
	sudo chmod +x ./kubectl
	sudo mv ./kubectl /usr/local/bin/kubectl

install:
ifeq ($(OS_NAME),Darwin)
	$(MAKE) _install_on_mac
else ifeq ($(OS_NAME),Linux)
	$(MAKE) _install_on_ubuntu
endif

run: _run_if_not_exists
	@$(MAKE) _apply

rm:
	@kubectl delete -f $(shell pwd)/k8s/admin/namespace.yml 2> /dev/null; \
	minikube stop

delete:
	@minikube delete 2> /dev/null

reapply:
	@kubectl delete -f $(shell pwd)/k8s/admin/namespace.yml
	@$(MAKE) _apply

pods:
	@kubectl -n dev get pods

db_bash:
	@kubectl -n dev exec -it $(shell $(POD_DB_NAME)) -- bash

dc_bash:
	@kubectl -n dev exec -it $(shell $(POD_DC_NAME)) -- bash

td_bash:
	@kubectl -n dev exec -it $(shell $(POD_TD_NAME)) -- bash