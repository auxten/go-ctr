package main

import (
	"github.com/gin-gonic/gin"
	"github.com/sashabaranov/go-fastapi"
)

type EchoInput struct {
	Phrase string `json:"phrase"`
}

type EchoOutput struct {
	OriginalInput EchoInput `json:"original_input"`
}

func EchoHandler(ctx *gin.Context, in EchoInput) (out EchoOutput, err error) {
	out.OriginalInput = in
	return
}

func main() {
	r := gin.Default()

	myRouter := fastapi.NewRouter()
	myRouter.AddCall("/echo", EchoHandler)

	r.POST("/api/*path", myRouter.GinHandler) // must have *path parameter
	r.Run()
}
