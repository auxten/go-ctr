package din

import (
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
