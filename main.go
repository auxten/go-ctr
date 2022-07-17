package main

import (
	"fmt"
	"regexp"
)

type If interface {
	Fn1() string
	Do() string
}

type IfImpl struct {
	name string
}

func (i IfImpl) Do() string {
	fmt.Printf("%s\n", i.Fn1())
	return i.name
}

func (i IfImpl) Fn1() string {
	return "fn1"
}

type IfImpl2 struct {
	IfImpl
}

func (i IfImpl2) Fn1() string {
	return "fn2"
}

func main() {
	im2 := IfImpl2{}
	im2.Do()
	yearRegex := regexp.MustCompile(`\((\d{4})\)`)
	fmt.Println(yearRegex.FindStringSubmatch("dfasd (2012)")[1])
}
