package din

import (
	log "github.com/sirupsen/logrus"
	G "gorgonia.org/gorgonia"
)

// PRelu is the slop learnable LeakyRelu activation function
// slop should be a scalar
func PRelu(x, slop *G.Node) (retVal *G.Node) {
	negative := G.Must(G.HadamardProd(G.Must(G.Sub(x, G.Must(G.Abs(x)))), slop))
	positive := G.Must(G.Add(x, G.Must(G.Abs(x))))
	retVal = G.Must(G.Mul(G.Must(G.Add(negative, positive)), G.NewConstant(0.5)))
	return
}

// EucDistance is the Euclidean distance between two matrix, typically used for
// calculating the distance between two embedding.
// Case1: x, y shapes are same, no broadcast. output shape will be x.shape[:-1]
// Case2: x, y shapes are different, broadcast will be applied on the smaller dim.
// output shape will be something like x.shape[:-1] but with a broadcast dim
func EucDistance(x, y *G.Node) (retVal *G.Node) {
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
		log.Panicf("x, y shapes not supported: %v, %v", x.Shape(), y.Shape())
	}

	retVal = G.Must(G.Sqrt(G.Must(G.Sum(G.Must(G.Square(sub)), x.Dims()-1))))
	return
}
