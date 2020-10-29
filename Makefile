CONTAINER_NAME=$(shell docker ps | grep binance_trader_v2:latest | cut -d ' ' -f 1)
CONTAINER_NAMES=$(shell docker ps -a | grep binance_trader_v2:latest | cut -d ' ' -f 1)

rm:
	docker rm -f $(CONTAINER_NAMES) || echo

run: rm
	docker run -d --gpus all -v $(PWD)/develop:/app binance_trader_v2:latest tail -f /dev/null

run_cpu: rm
	docker run -d -v $(PWD)/develop:/app binance_trader_v2:latest tail -f /dev/null

bash:
	docker exec -it $(CONTAINER_NAME) bash

build_container:
	docker build develop/dockerfiles -t binance_trader_v2:latest

build_rawdata:
	python -m rawdata_builder.build_rawdata $(ARGS)

build_dataset:
	python -m dataset_builder.build_dataset_v1 $(ARGS)

train:
	python -m trainer.models.predictor_v1 $(ARGS)

review:
	python -m reviewer.reviewer_v1 $(ARGS)