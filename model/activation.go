package model

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// PRelu32 is the slop learnable LeakyRelu activation function
// slop should be a scalar
func PRelu32(x, slop *G.Node) (retVal *G.Node) {
	negative := G.Must(G.HadamardProd(G.Must(G.Sub(x, G.Must(G.Abs(x)))), slop))
	positive := G.Must(G.Add(x, G.Must(G.Abs(x))))
	retVal = G.Must(G.Mul(G.Must(G.Add(negative, positive)), G.NewConstant(float32(0.5))))
	return
}

// EucDistance is the Euclidean distance between two matrix, typically used for
// calculating the distance between two embedding.
// Case1: x, y shapes are same, no broadcast. output shape will be x.shape[:-1]
// Case2: x, y shapes are different, broadcast will be applied on the smaller dim.
// output shape will be something like x.shape[:-1] but with a broadcast dim
func EucDistance(x, y *G.Node) (retVal *G.Node, err error) {
	var sub *G.Node

	if x.Dims() == y.Dims() {
		var sameShape = true
		for i := 0; i < x.Dims(); i++ {
			if x.Shape()[i] != y.Shape()[i] {
				if x.Shape()[i] < y.Shape()[i] {
					sub = G.Must(G.BroadcastSub(x, y, []byte{byte(i)}, nil))
				} else {
					sub = G.Must(G.BroadcastSub(x, y, nil, []byte{byte(i)}))
				}
				sameShape = false
				break
			}
		}
		if sameShape && sub == nil {
			sub = G.Must(G.Sub(x, y))
		}
	}
	if sub == nil {
		err = fmt.Errorf("x, y shapes not supported: %v, %v", x.Shape(), y.Shape())
		return
	}

	retVal = G.Must(G.Sqrt(G.Must(G.Sum(G.Must(G.Square(sub)), x.Dims()-1))))
	return
}

// CosineDistance is the cosine distance between two matrix, typically used for
// calculating the distance between two embedding.
// Case1: x, y shapes are same, no broadcast. output shape will be x.shape[:-1]
// Case2: x, y shapes are different, broadcast will be applied on the smaller dim.
// output shape will be something like x.shape[:-1] but with a broadcast dim
func CosineDistance(x, y *G.Node) (retVal *G.Node, err error) {
	if x.Dims() != y.Dims() {
		err = fmt.Errorf("x, y shapes not supported: %v, %v", x.Shape(), y.Shape())
		return
	}

	for i := 0; i < x.Dims(); i++ {
		if x.Shape()[i] != y.Shape()[i] {
			if x.Shape()[i] < y.Shape()[i] {
				x, y, err = G.Broadcast(x, y, G.NewBroadcastPattern([]byte{byte(i)}, nil))
			} else {
				x, y, err = G.Broadcast(x, y, G.NewBroadcastPattern(nil, []byte{byte(i)}))
			}
			if err != nil {
				return
			}
		}
	}
	xNorm := G.Must(G.Sqrt(G.Must(G.Sum(G.Must(G.Square(x)), x.Dims()-1))))
	yNorm := G.Must(G.Sqrt(G.Must(G.Sum(G.Must(G.Square(y)), y.Dims()-1))))

	retVal = G.Must(G.HadamardDiv(
		G.Must(G.Sum(G.Must(G.HadamardProd(x, y)), x.Dims()-1)),
		G.Must(G.HadamardProd(xNorm, yNorm)),
	))
	return
}
