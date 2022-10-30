package din

import (
	"encoding/json"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type SimpleMLP struct {
	g  *G.ExprGraph
	vm G.VM

	uProfileDim, uBehaviorSize, uBehaviorDim int
	iFeatureDim                              int
	cFeatureDim                              int

	//input nodes
	xUserProfile, xUbMatrix, xItemFeature, xCtxFeature *G.Node
	//learnable nodes
	mlp0, mlp1, mlp2 *G.Node
	d0, d1           float64 // dropout probabilities
	out              *G.Node
}

func (mlp *SimpleMLP) In() G.Nodes {
	return G.Nodes{mlp.xUserProfile, mlp.xUbMatrix, mlp.xItemFeature, mlp.xCtxFeature}
}

type mlpModel struct {
	UProfileDim   int       `json:"uProfileDim"`
	UBehaviorSize int       `json:"uBehaviorSize"`
	UBehaviorDim  int       `json:"uBehaviorDim"`
	IFeatureDim   int       `json:"iFeatureDim"`
	CFeatureDim   int       `json:"cFeatureDim"`
	Mlp0          []float64 `json:"mlp0"`
	Mlp1          []float64 `json:"mlp1"`
	Mlp2          []float64 `json:"mlp2"`
}

func (mlp *SimpleMLP) Marshal() (data []byte, err error) {
	model := mlpModel{
		UProfileDim:   mlp.uProfileDim,
		UBehaviorSize: mlp.uBehaviorSize,
		UBehaviorDim:  mlp.uBehaviorDim,
		IFeatureDim:   mlp.iFeatureDim,
		CFeatureDim:   mlp.cFeatureDim,
		Mlp0:          mlp.mlp0.Value().Data().([]float64),
		Mlp1:          mlp.mlp1.Value().Data().([]float64),
		Mlp2:          mlp.mlp2.Value().Data().([]float64),
	}
	return json.Marshal(model)
}

func NewSimpleMLPFromJson(data []byte) (mlp *SimpleMLP, err error) {
	var m mlpModel
	if err = json.Unmarshal(data, &m); err != nil {
		return
	}
	var (
		g             = G.NewGraph()
		uProfileDim   = m.UProfileDim
		uBehaviorSize = m.UBehaviorSize
		uBehaviorDim  = m.UBehaviorDim
		iFeatureDim   = m.IFeatureDim
		cFeatureDim   = m.CFeatureDim
		mlp0_0        = uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim + cFeatureDim
	)

	mlp0 := G.NewMatrix(g, dt,
		G.WithShape(mlp0_0, mlp0_1),
		G.WithName("mlp0"),
		G.WithValue(tensor.New(tensor.WithShape(mlp0_0, mlp0_1), tensor.WithBacking(m.Mlp0))),
	)

	mlp1 := G.NewMatrix(g, dt,
		G.WithShape(mlp0_1, mlp1_2),
		G.WithName("mlp1"),
		G.WithValue(tensor.New(tensor.WithShape(mlp0_1, mlp1_2), tensor.WithBacking(m.Mlp1))),
	)

	mlp2 := G.NewMatrix(g, dt,
		G.WithShape(mlp1_2, 1),
		G.WithName("mlp2"),
		G.WithValue(tensor.New(tensor.WithShape(mlp1_2, 1), tensor.WithBacking(m.Mlp2))),
	)

	mlp = &SimpleMLP{
		uProfileDim:   uProfileDim,
		uBehaviorSize: uBehaviorSize,
		uBehaviorDim:  uBehaviorDim,
		iFeatureDim:   iFeatureDim,
		cFeatureDim:   cFeatureDim,
		g:             g,
		mlp0:          mlp0,
		mlp1:          mlp1,
		mlp2:          mlp2,
	}

	return
}

func (mlp *SimpleMLP) Vm() G.VM {
	return mlp.vm
}

func (mlp *SimpleMLP) SetVM(vm G.VM) {
	mlp.vm = vm
}

func NewSimpleMLP(
	uProfileDim, uBehaviorSize, uBehaviorDim int,
	iFeatureDim int,
	cFeatureDim int,
) (mlp *SimpleMLP) {
	g := G.NewGraph()
	mlp0 := G.NewMatrix(g, G.Float64, G.WithShape(uProfileDim+uBehaviorSize*uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1), G.WithName("mlp0"), G.WithInit(G.HeN(1.0)))
	mlp1 := G.NewMatrix(g, G.Float64, G.WithShape(mlp0_1, mlp1_2), G.WithName("mlp1"), G.WithInit(G.HeN(1.0)))
	mlp2 := G.NewMatrix(g, G.Float64, G.WithShape(mlp1_2, 1), G.WithName("mlp2"), G.WithInit(G.HeN(1.0)))
	return &SimpleMLP{
		uProfileDim:   uProfileDim,
		uBehaviorSize: uBehaviorSize,
		uBehaviorDim:  uBehaviorDim,
		iFeatureDim:   iFeatureDim,
		cFeatureDim:   cFeatureDim,

		g:    g,
		d0:   0.003,
		d1:   0.003,
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
	xUserBehaviors := G.Must(G.Reshape(ubMatrix, tensor.Shape{batchSize, uBehaviorSize * uBehaviorDim}))
	// item feature
	// context feature
	// concat
	x := G.Must(G.Concat(1, xUserProfile, xUserBehaviors, xItemFeature, xCtxFeature))
	// mlp
	mlp0Out := G.Must(G.Sigmoid(G.Must(G.Mul(x, mlp.mlp0))))
	mlp0Out = G.Must(G.Dropout(mlp0Out, mlp.d0))
	mlp1Out := G.Must(G.Sigmoid(G.Must(G.Mul(mlp0Out, mlp.mlp1))))
	mlp1Out = G.Must(G.Dropout(mlp1Out, mlp.d1))

	mlp.out = G.Must(G.Sigmoid(G.Must(G.Mul(mlp1Out, mlp.mlp2))))
	mlp.xUserProfile = xUserProfile
	mlp.xItemFeature = xItemFeature
	mlp.xCtxFeature = xCtxFeature
	mlp.xUbMatrix = ubMatrix

	return
}
