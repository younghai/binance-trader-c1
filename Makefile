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