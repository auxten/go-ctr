package din

import (
	"encoding/json"
	"fmt"
	_ "net/http/pprof"

	"github.com/pkg/errors"
	log "github.com/sirupsen/logrus"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type DinNet struct {
	uProfileDim, uBehaviorSize, uBehaviorDim int
	iFeatureDim                              int
	cFeatureDim                              int

	g *G.ExprGraph

	vm G.VM

	//input nodes
	xUserProfile, xUbMatrix, xItemFeature, xCtxFeature *G.Node

	mlp0, mlp1, mlp2 *G.Node // weights of MLP layers
	d0, d1           float64 // dropout probabilities
	att0, att1       *G.Node // weights of Attention layers

	out *G.Node
}

type dinModel struct {
	UProfileDim   int         `json:"uProfileDim"`
	UBehaviorSize int         `json:"uBehaviorSize"`
	UBehaviorDim  int         `json:"uBehaviorDim"`
	IFeatureDim   int         `json:"iFeatureDim"`
	CFeatureDim   int         `json:"cFeatureDim"`
	Mlp0          []float64   `json:"mlp0"`
	Mlp1          []float64   `json:"mlp1"`
	Mlp2          []float64   `json:"mlp2"`
	Att0          [][]float64 `json:"att0"`
	Att1          [][]float64 `json:"att1"`
}

func (din *DinNet) Vm() G.VM {
	return din.vm
}

func (din *DinNet) SetVM(vm G.VM) {
	din.vm = vm
}

func (din *DinNet) Marshal() (data []byte, err error) {
	modelData := dinModel{
		UProfileDim:   din.uProfileDim,
		UBehaviorSize: din.uBehaviorSize,
		UBehaviorDim:  din.uBehaviorDim,
		IFeatureDim:   din.iFeatureDim,
		CFeatureDim:   din.cFeatureDim,
		Mlp0:          din.mlp0.Value().Data().([]float64),
		Mlp1:          din.mlp1.Value().Data().([]float64),
		Mlp2:          din.mlp2.Value().Data().([]float64),
	}
	modelData.Att0 = make([][]float64, din.uBehaviorSize)
	modelData.Att1 = make([][]float64, din.uBehaviorSize)
	for i := 0; i < din.uBehaviorSize; i++ {
		modelData.Att0[i] = din.att0[i].Value().Data().([]float64)
		modelData.Att1[i] = din.att1[i].Value().Data().([]float64)
	}
	//marshal to json
	data, err = json.Marshal(modelData)

	return
}

func NewDinNetFromJson(data []byte) (din *DinNet, err error) {
	var m dinModel
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
		att0_0        = uBehaviorDim + iFeatureDim + uBehaviorSize
	)

	// attention layer
	att0 := make([]*G.Node, m.UBehaviorSize)
	att1 := make([]*G.Node, m.UBehaviorSize)
	for i := 0; i < m.UBehaviorSize; i++ {
		att0[i] = G.NewMatrix(
			g,
			dt,
			G.WithShape(att0_0, att0_1),
			G.WithValue(tensor.New(tensor.WithShape(att0_0, att0_1), tensor.WithBacking(m.Att0[i]))),
			G.WithName(fmt.Sprintf("att0-%d", i)),
		)
		att1[i] = G.NewMatrix(
			g,
			dt,
			G.WithShape(att0_1, 1),
			G.WithValue(tensor.New(tensor.WithShape(att0_1, 1), tensor.WithBacking(m.Att1[i]))),
			G.WithName(fmt.Sprintf("att1-%d", i)),
		)
	}
	mlp0 := G.NewMatrix(g, dt,
		G.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1),
		G.WithName("mlp0"),
		G.WithValue(tensor.New(
			tensor.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1),
			tensor.WithBacking(m.Mlp0)),
		),
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

	din = &DinNet{
		uProfileDim:   m.UProfileDim,
		uBehaviorSize: m.UBehaviorSize,
		uBehaviorDim:  m.UBehaviorDim,
		iFeatureDim:   m.IFeatureDim,
		cFeatureDim:   m.CFeatureDim,
		g:             g,
		att0:          att0,
		att1:          att1,
		mlp0:          mlp0,
		mlp1:          mlp1,
		mlp2:          mlp2,
	}
	return
}

func (din *DinNet) Graph() *G.ExprGraph {
	return din.g
}

func (din *DinNet) Out() *G.Node {
	return din.out
}

func (din *DinNet) In() G.Nodes {
	return G.Nodes{din.xUserProfile, din.xUbMatrix, din.xItemFeature, din.xCtxFeature}
}

func (din *DinNet) learnable() G.Nodes {
	ret := make(G.Nodes, 3, 3+2*din.uBehaviorSize)
	ret[0] = din.mlp0
	ret[1] = din.mlp1
	ret[2] = din.mlp2
	ret = append(ret, din.att0...)
	ret = append(ret, din.att1...)
	return ret
}

func NewDinNet(
	uProfileDim, uBehaviorSize, uBehaviorDim int,
	iFeatureDim int,
	cFeatureDim int,
) *DinNet {
	if uBehaviorDim != iFeatureDim {
		log.Fatalf("uBehaviorDim %d != iFeatureDim %d", uBehaviorDim, iFeatureDim)
	}
	g := G.NewGraph()
	// attention layer
	att0 := make([]*G.Node, uBehaviorSize)
	att1 := make([]*G.Node, uBehaviorSize)
	for i := 0; i < uBehaviorSize; i++ {
		att0[i] = G.NewTensor(g, dt, 2, G.WithShape(uBehaviorDim+iFeatureDim+uBehaviorSize*uBehaviorDim, att0_1), G.WithName(fmt.Sprintf("att0-%d", i)), G.WithInit(G.Gaussian(0, 1)))
		att1[i] = G.NewTensor(g, dt, 2, G.WithShape(att0_1, 1), G.WithName(fmt.Sprintf("att1-%d", i)), G.WithInit(G.Gaussian(0, 1)))
	}

	// user behaviors are represented as a sequence of item embeddings. Before
	// being fed into the MLP, we need to flatten the sequence into a single with
	// sum pooling with Attention as the weights which is the key point of DIN model.
	mlp0 := G.NewMatrix(g, dt, G.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1), G.WithName("mlp0"), G.WithInit(G.Gaussian(0, 1)))

	mlp1 := G.NewMatrix(g, dt, G.WithShape(mlp0_1, mlp1_2), G.WithName("mlp1"), G.WithInit(G.Gaussian(0, 1)))

	mlp2 := G.NewMatrix(g, dt, G.WithShape(mlp1_2, 1), G.WithName("mlp2"), G.WithInit(G.Gaussian(0, 1)))

	return &DinNet{
		uProfileDim:   uProfileDim,
		uBehaviorSize: uBehaviorSize,
		uBehaviorDim:  uBehaviorDim,
		iFeatureDim:   iFeatureDim,
		cFeatureDim:   cFeatureDim,

		g:    g,
		att0: att0,
		att1: att1,

		d0: 0.005,
		d1: 0.005,

		mlp0: mlp0,
		mlp1: mlp1,
		mlp2: mlp2,
	}
}

// Fwd performs the forward pass
// xUserProfile: [batchSize, userProfileDim]
// xUbMatrix: [batchSize, uBehaviorSize* uBehaviorDim]
// xUserBehaviors: [batchSize, uBehaviorSize, uBehaviorDim]
// xItemFeature: [batchSize, iFeatureDim]
// xContextFeature: [batchSize, cFeatureDim]
func (din *DinNet) Fwd(xUserProfile, xUbMatrix, xItemFeature, xCtxFeature *G.Node, batchSize, uBehaviorSize, uBehaviorDim int) (err error) {
	iFeatureDim := xItemFeature.Shape()[1]
	if uBehaviorDim != iFeatureDim {
		return errors.Errorf("uBehaviorDim %d != iFeatureDim %d", uBehaviorDim, iFeatureDim)
	}
	xUserBehaviors := G.Must(G.Reshape(xUbMatrix, tensor.Shape{batchSize, uBehaviorSize, uBehaviorDim}))
	xItemFeature3d := G.Must(G.Reshape(xItemFeature, tensor.Shape{batchSize, 1, iFeatureDim}))

	// attention layer

	// euclideanDistance: [batchSize, uBehaviorSize]
	euclideanDistance := EucDistance(xUserBehaviors, xItemFeature3d)

	//// outProduct should computed batch by batch!!!!
	//outProdVecs := make([]*G.Node, batchSize)
	//for i := 0; i < batchSize; i++ {
	//	// ubVec.Shape() = [uBehaviorSize * uBehaviorDim]
	//	ubVec := G.Must(G.Slice(xUbMatrix, G.S(i)))
	//	// item.Shape() = [iFeatureDim]
	//	itemVec := G.Must(G.Slice(xItemFeature, G.S(i)))
	//	// outProd.Shape() = [uBehaviorSize * uBehaviorDim, iFeatureDim]
	//	outProd := G.Must(G.OuterProd(ubVec, itemVec))
	//	outProdVecs[i] = G.Must(G.Reshape(outProd, tensor.Shape{uBehaviorSize * uBehaviorDim * iFeatureDim}))
	//}
	////outProductsVec.Shape() = [batchSize * uBehaviorSize * uBehaviorDim * iFeatureDim]
	//outProductsVec := G.Must(G.Concat(0, outProdVecs...))

	//outProducts := G.Must(G.Reshape(outProductsVec, tensor.Shape{batchSize, uBehaviorSize * uBehaviorDim * iFeatureDim}))

	actOuts := G.NewTensor(din.Graph(), dt, 2, G.WithShape(batchSize, uBehaviorDim), G.WithName("actOuts"), G.WithInit(G.Zeroes()))
	for i := 0; i < uBehaviorSize; i++ {
		// xUserBehaviors[:, i, :], ub.shape: [batchSize, uBehaviorDim]
		ub := G.Must(G.Slice(xUserBehaviors, []tensor.Slice{nil, G.S(i)}...))
		// Concat all xUserBehaviors[i], outProducts, xItemFeature
		// actConcat.Shape() = [batchSize, uBehaviorDim+iFeatureDim+uBehaviorSize]
		actConcat := G.Must(G.Concat(1, ub, euclideanDistance, xItemFeature))
		actOut := G.Must(G.BroadcastHadamardProd(
			ub,
			G.Must(G.Sigmoid(
				G.Must(G.Mul(
					G.Must(G.Mul(actConcat, din.att0[i])),
					din.att1[i],
				)))), // [batchSize, 1]
			nil, []byte{1},
		)) // [batchSize, uBehaviorDim]

		// Sum pooling
		actOuts = G.Must(G.Add(actOuts, actOut))
	}

	// Concat all xUserProfile, actOuts, xItemFeature, xCtxFeature
	concat := G.Must(G.Concat(1, xUserProfile, actOuts, xItemFeature, xCtxFeature))

	// MLP

	// mlp0.Shape: [userProfileDim+userBehaviorDim+itemFeatureDim+contextFeatureDim, 200]
	// out.Shape: [batchSize, 200]
	mlp0Out := G.Must(G.Sigmoid(G.Must(G.Mul(concat, din.mlp0))))
	mlp0Out = G.Must(G.Dropout(mlp0Out, din.d0))
	// mlp1.Shape: [200, 80]
	// out.Shape: [batchSize, 80]
	mlp1Out := G.Must(G.Sigmoid(G.Must(G.Mul(mlp0Out, din.mlp1))))
	mlp1Out = G.Must(G.Dropout(mlp1Out, din.d1))
	// mlp2.Shape: [80, 1]
	// out.Shape: [batchSize, 1]
	mlp2Out := G.Must(G.Sigmoid(G.Must(G.Mul(mlp1Out, din.mlp2))))

	din.out = mlp2Out
	din.xUserProfile = xUserProfile
	din.xItemFeature = xItemFeature
	din.xCtxFeature = xCtxFeature
	din.xUbMatrix = xUbMatrix
	return
}
