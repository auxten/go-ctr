package din

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type SimpleMLP struct {
	g                *G.ExprGraph
	mlp0, mlp1, mlp2 *G.Node
	d0, d1           float64 // dropout probabilities
	out              *G.Node
}

func NewSimpleMLP(g *G.ExprGraph,
	uProfileDim, uBehaviorSize, uBehaviorDim int,
	iFeatureDim int,
	ctxFeatureDim int,
) (mlp *SimpleMLP) {
	mlp0 := G.NewMatrix(g, G.Float64, G.WithShape(uProfileDim+uBehaviorSize*uBehaviorDim+iFeatureDim+ctxFeatureDim, 200), G.WithName("mlp0"), G.WithInit(G.Gaussian(0, 1)))
	mlp1 := G.NewMatrix(g, G.Float64, G.WithShape(200, 80), G.WithName("mlp1"), G.WithInit(G.Gaussian(0, 1)))
	mlp2 := G.NewMatrix(g, G.Float64, G.WithShape(80, 1), G.WithName("mlp2"), G.WithInit(G.Gaussian(0, 1)))
	return &SimpleMLP{
		g:    g,
		d0:   0.01,
		d1:   0.01,
		mlp0: mlp0,
		mlp1: mlp1,
		mlp2: mlp2,
	}
}

func (mlp *SimpleMLP) Graph() *G.ExprGraph {
	return mlp.g
}

func (mlp *SimpleMLP) Out() *G.Node {
	return mlp.out
}

func (mlp *SimpleMLP) learnable() G.Nodes {
	return G.Nodes{mlp.mlp0, mlp.mlp1, mlp.mlp2}
}

func (mlp *SimpleMLP) Fwd(xUserProfile, ubMatrix, xItemFeature, xCtxFeature *G.Node, batchSize, uBehaviorSize, uBehaviorDim int) (err error) {
	// user behaviors
	ubMatrix = G.Must(G.Reshape(ubMatrix, tensor.Shape{batchSize, uBehaviorSize * uBehaviorDim}))
	// item feature
	// context feature
	// concat
	x := G.Must(G.Concat(1, xUserProfile, ubMatrix, xItemFeature, xCtxFeature))
	// mlp
	mlp0Out := G.Must(G.LeakyRelu(G.Must(G.Mul(x, mlp.mlp0)), 0.1))
	mlp0Out = G.Must(G.Dropout(mlp0Out, mlp.d0))
	mlp1Out := G.Must(G.LeakyRelu(G.Must(G.Mul(mlp0Out, mlp.mlp1)), 0.1))
	mlp1Out = G.Must(G.Dropout(mlp1Out, mlp.d1))
	mlp.out = G.Must(G.Sigmoid(G.Must(G.Mul(mlp1Out, mlp.mlp2))))

	return
}
