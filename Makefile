.PHONY: lint build build-frontend

default: build

## run lint to format golang code
lint:
	gofmt -w -s ./
	goimports -local github.com/auxten/edgeRec -w ./

.PHONY: build-frontend
## build frontend
build-frontend:
	cd frontend && pnpm run bootstrap
	cd frontend && pnpm run build

.PHONY: build
## build golang backend
build: build-frontend
	go build -o edgeRec main.go
