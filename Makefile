.PHONY: lint
## run lint to format golang code
lint:
	gofmt -w -s ./
	goimports -local github.com/auxten/edgeRec -w ./
