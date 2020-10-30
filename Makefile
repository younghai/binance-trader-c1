CONTAINER_NAME=$(shell docker ps | grep binance_trader_v2:latest | cut -d ' ' -f 1)
CONTAINER_NAMES=$(shell docker ps -a | grep binance_trader_v2:latest | cut -d ' ' -f 1)

build_container:
	docker build develop/dockerfiles -t binance_trader_v2:latest

rm:
	docker rm -f $(CONTAINER_NAMES) || echo

run_cpu: rm
	docker run -d -v $(PWD)/develop:/app binance_trader_v2:latest tail -f /dev/null

run: rm
	docker run -d --gpus all -v $(PWD)/develop:/app binance_trader_v2:latest tail -f /dev/null || make run_cpu

bash:
	docker exec -it $(CONTAINER_NAME) bash

download_kaggle_data:
ifeq ($(CONTAINER_NAME),)
	make run
endif
	docker exec -it `docker ps | grep binance_trader_v2:latest | cut -d ' ' -f 1` python -m rawdata_builder.download_kaggle_data $(ARGS)

build_rawdata:
ifeq ($(CONTAINER_NAME),)
	make run
endif
	docker exec -it `docker ps | grep binance_trader_v2:latest | cut -d ' ' -f 1` python -m rawdata_builder.build_rawdata $(ARGS)

build_dataset:
ifeq ($(CONTAINER_NAME),)
	make run
endif
	docker exec -it `docker ps | grep binance_trader_v2:latest | cut -d ' ' -f 1` python -m dataset_builder.build_dataset_v1 $(ARGS)

train:
ifeq ($(CONTAINER_NAME),)
	make run
endif
	docker exec -it `docker ps | grep binance_trader_v2:latest | cut -d ' ' -f 1` python -m trainer.models.predictor_v1 $(ARGS)

review:
ifeq ($(CONTAINER_NAME),)
	make run
endif
	docker exec -it `docker ps | grep binance_trader_v2:latest | cut -d ' ' -f 1` python -m reviewer.reviewer_v1 $(ARGS)