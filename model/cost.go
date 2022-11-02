package model

import (
	G "gorgonia.org/gorgonia"
)

// BinaryCrossEntropy32 calculates the binary cross entropy cost
// loss formula: -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)
func BinaryCrossEntropy32(yPred, yTrue *G.Node) *G.Node {
	positive := G.Must(G.HadamardProd(G.Must(G.Log(yPred)), yTrue))
	negative := G.Must(G.HadamardProd(G.Must(
		G.Log(G.Must(G.Sub(G.NewConstant(float32(1.0+1e-8)), yPred)))),
		G.Must(G.Sub(G.NewConstant(float32(1.0)), yTrue)),
	))
	cost := G.Must(G.Neg(G.Must(G.Mean(G.Must(G.Add(positive, negative))))))
	return cost
}

// MSE32 calculates the Mean Squared Error cost
func MSE32(yPred, yTrue *G.Node) *G.Node {
	cost := G.Must(G.Mean(G.Must(G.Square(G.Must(G.Sub(yPred, yTrue))))))
	return cost
}

// RMS32 calculates the Root Mean Squared error cost
func RMS32(yPred, yTrue *G.Node) *G.Node {
	cost := G.Must(G.Sqrt(G.Must(G.Mean(G.Must(G.Square(G.Must(G.Sub(yPred, yTrue))))))))
	return cost
}
