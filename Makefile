.PHONY: lint build build-frontend

default: build
commit := $(shell git describe --match= --always --dirty)
version := $(shell git describe --tags --always --dirty)

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
	CGO_ENABLED=1 go build -o edgeRec main.go

## build golang backend
release-build: build-frontend
	CGO_ENABLED=1 go build -ldflags=" -X main.Version=$(version) -X main.Commit=$(commit)" -o "edgeRec_$$(go env GOOS)_$$(go env GOARCH)" main.go
