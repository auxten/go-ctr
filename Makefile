.PHONY: lint
## run lint to format golang code
lint:
	gofmt -w -s ./
	goimports -local github.com/auxten/edgeRec -w ./

## build frontend
build-frontend:
	cd frontend && pnpm run bootstrap
	cd frontend && pnpm run build

## build golang backend
build: build-frontend
	go build -o edgeRec main.go
