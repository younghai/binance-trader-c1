GPU_TYPE=NVIDIA
CONTAINER_NAME=docker ps | grep binance_trader_develop:latest | cut -d ' ' -f 1
CONTAINER_NAMES=docker ps -a | grep binance_trader_develop:latest | cut -d ' ' -f 1

print:
	echo $(GPU_TYPE)

_build_container:
ifeq ($(shell echo $(GPU_TYPE) | tr [a-z] [A-Z]),NVIDIA)
	@docker build dockerfiles -t binance_trader_develop:latest --build-arg BASE_IMAGE=pytorch/pytorch:latest
else ifeq ($(shell echo $(GPU_TYPE) | tr [a-z] [A-Z]),AMD)
	@docker build dockerfiles -t amd_binance_trader_develop:latest --build-arg BASE_IMAGE=rocm/pytorch:latest
else
	$(error Unexpected GPU_TYPE: $(GPU_TYPE))
endif

_run_if_not_exists:
ifeq ($(shell $(CONTAINER_NAME)),)
	$(MAKE) run
endif
	@echo

rm:
	@docker rm -f $(shell $(CONTAINER_NAMES)) 2> /dev/null || echo

run: _build_container rm
ifeq ($(shell echo $(GPU_TYPE) | tr [a-z] [A-Z]),NVIDIA)
	@docker run -d --gpus all --shm-size=16gb -v $(shell pwd):/app binance_trader_develop:latest tail -f /dev/null 2> /dev/null || ($(MAKE) rm ; docker run -d --shm-size=16gb -v $(shell pwd):/app binance_trader_develop:latest tail -f /dev/null)
else ifeq ($(shell echo $(GPU_TYPE) | tr [a-z] [A-Z]),AMD)
	@docker run -d --network=host --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=16gb -v $(shell pwd):/app amd_binance_trader_develop:latest tail -f /dev/null
else
	$(error Unexpected GPU_TYPE: $(GPU_TYPE))
endif

bash: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) bash

download_kaggle_data: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) python -m rawdata_builder.download_kaggle_data $(ARGS)

build_rawdata: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) python -m rawdata_builder.build_rawdata $(ARGS)

build_dataset: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) python -m dataset_builder.build_dataset_v1 $(ARGS)

train: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) python -m trainer.models.predictor_v1 train --mode=train $(ARGS)

generate: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) python -m trainer.models.predictor_v1 generate --mode=test $(ARGS)

review: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) python -m reviewer.reviewer_v1 run --in_shell True $(ARGS)

display_review: _run_if_not_exists
	docker exec -it $(shell $(CONTAINER_NAME)) python -m reviewer.reviewer_v1 display --in_shell True $(ARGS)