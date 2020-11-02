DB_CONTAINER_NAME=docker ps | grep postgres:latest | cut -d ' ' -f 1
DB_CONTAINER_NAMES=docker ps -a | grep postgres:latest | cut -d ' ' -f 1

rm_db:
	@docker rm -f $(shell $(DB_CONTAINER_NAMES)) 2> /dev/null || echo

run_db: rm_db
	@docker run -d --shm-size=1gb -p 5432:5432 --name db -v $(shell pwd)/storage/db:/var/lib/postgresql/data -e POSTGRES_PASSWORD=password postgres:latest

run_db_if_not_exists:
ifeq ($(shell $(DB_CONTAINER_NAME)),)
	make run_db
endif

up_db: run_db_if_not_exists
	docker exec -it $(shell $(DB_CONTAINER_NAME)) bash -c \
	'psql -U postgres -d postgres -c \
	"create table if not exists pricing (\
		id serial primary key, \
		open float not null, \
		high float not null, \
		low float not null, \
		close float not null, \
		volume float not null, \
		target_on timestamp not null, \
		created_on timestamp not null \
	)"'

bash_db: run_db_if_not_exists
	docker exec -it $(shell $(DB_CONTAINER_NAME)) bash
