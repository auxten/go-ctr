package din

import (
	"encoding/json"
	_ "net/http/pprof"

	"github.com/auxten/go-ctr/model"
	"github.com/pkg/errors"
	log "github.com/sirupsen/logrus"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	// magic numbers for din paper
	att0_1 = 36
	mlp0_1 = 200
	mlp1_2 = 80
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
	d0, d1           float32 // dropout probabilities
	att0             *G.Node // weights of attention layer
	//att1       *G.Node // weights of Attention layers

	out *G.Node
}

type dinModel struct {
	UProfileDim   int       `json:"uProfileDim"`
	UBehaviorSize int       `json:"uBehaviorSize"`
	UBehaviorDim  int       `json:"uBehaviorDim"`
	IFeatureDim   int       `json:"iFeatureDim"`
	CFeatureDim   int       `json:"cFeatureDim"`
	Mlp0          []float32 `json:"mlp0"`
	Mlp1          []float32 `json:"mlp1"`
	Mlp2          []float32 `json:"mlp2"`
	Att0          []float32 `json:"att0"`
	//Att1          []float32 `json:"att1"`
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
		Mlp0:          din.mlp0.Value().Data().([]float32),
		Mlp1:          din.mlp1.Value().Data().([]float32),
		Mlp2:          din.mlp2.Value().Data().([]float32),
		Att0:          din.att0.Value().Data().([]float32),
		//Att1:          din.att1.Value().Data().([]float32),
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
	)

	// attention layer
	att0 := G.NewMatrix(
		g,
		model.DT,
		G.WithShape(1, uBehaviorSize),
		G.WithValue(tensor.New(tensor.WithShape(1, uBehaviorSize), tensor.WithBacking(m.Att0))),
		G.WithName("att0"),
	)
	//att1 := G.NewMatrix(
	//	g,
	//	model.DT,
	//	G.WithShape(att0_1, 1),
	//	G.WithValue(tensor.New(tensor.WithShape(att0_1, 1), tensor.WithBacking(m.Att1[i]))),
	//	G.WithName("att1"),
	//)

	mlp0 := G.NewMatrix(g, model.DT,
		G.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1),
		G.WithName("mlp0"),
		G.WithValue(tensor.New(
			tensor.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1),
			tensor.WithBacking(m.Mlp0)),
		),
	)

	mlp1 := G.NewMatrix(g, model.DT,
		G.WithShape(mlp0_1, mlp1_2),
		G.WithName("mlp1"),
		G.WithValue(tensor.New(tensor.WithShape(mlp0_1, mlp1_2), tensor.WithBacking(m.Mlp1))),
	)

	mlp2 := G.NewMatrix(g, model.DT,
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
		//att1:          att1,
		mlp0: mlp0,
		mlp1: mlp1,
		mlp2: mlp2,
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

func (din *DinNet) Learnable() G.Nodes {
	ret := make(G.Nodes, 3, 3+2)
	ret[0] = din.mlp0
	ret[1] = din.mlp1
	ret[2] = din.mlp2
	ret = append(ret, din.att0)
	//ret = append(ret, din.att1)
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
	att0 := G.NewTensor(g, model.DT, 2, G.WithShape(1, uBehaviorSize), G.WithName("att0"), G.WithInit(G.ValuesOf(float32(1.0))))
	//att1 := G.NewTensor(g, model.DT, 3, G.WithShape(uBehaviorSize, att0_1, 1), G.WithName("att1"), G.WithInit(G.Gaussian(0, 1.0)))

	// user behaviors are represented as a sequence of item embeddings. Before
	// being fed into the MLP, we need to flatten the sequence into a single with
	// sum pooling with Attention as the weights which is the key point of DIN model.
	mlp0 := G.NewMatrix(g, model.DT, G.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1), G.WithName("mlp0"), G.WithInit(G.Gaussian(0, 1.0)))

	mlp1 := G.NewMatrix(g, model.DT, G.WithShape(mlp0_1, mlp1_2), G.WithName("mlp1"), G.WithInit(G.Gaussian(0, 1.0)))

	mlp2 := G.NewMatrix(g, model.DT, G.WithShape(mlp1_2, 1), G.WithName("mlp2"), G.WithInit(G.Gaussian(0, 1.0)))

	return &DinNet{
		uProfileDim:   uProfileDim,
		uBehaviorSize: uBehaviorSize,
		uBehaviorDim:  uBehaviorDim,
		iFeatureDim:   iFeatureDim,
		cFeatureDim:   cFeatureDim,

		g:    g,
		att0: att0,
		//att1: att1,

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

	// weight: [batchSize, uBehaviorSize]
	//weight := G.Must(G.Sub(G.NewConstant(float32(1.0)), G.Must(model.EucDistance(xUserBehaviors, xItemFeature3d))))
	weight := G.Must(G.Div(
		G.Must(G.Add(
			G.Must(model.CosineSimilarity(xUserBehaviors, xItemFeature3d)),
			G.NewConstant(float32(1.0)),
		)),
		G.NewConstant(float32(2.0)),
	))
	//euclideanDistance3d := G.Must(G.Reshape(distance, tensor.Shape{batchSize, uBehaviorSize, 1}))

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

	//xItemFeatureBroad.Shape() = [batchSize, uBehaviorSize, iFeatureDim]
	//xItemFeatureBroad, _, err := G.Broadcast(xItemFeature3d, xUserBehaviors, G.NewBroadcastPattern([]byte{1}, nil))
	//if err != nil {
	//	return errors.Wrap(err, "Broadcast")
	//}
	////actConcat.Shape() = [batchSize, uBehaviorSize, uBehaviorDim+1+iFeatureDim]
	//actConcat := G.Must(G.Concat(2, xUserBehaviors, euclideanDistance3d, xItemFeatureBroad))
	//actConcat3d := G.Must(G.Reshape(actConcat, tensor.Shape{batchSize, uBehaviorSize * (uBehaviorDim + 1 + iFeatureDim), 1}))

	//actOuts.Shape() = [batchSize, uBehaviorSize, uBehaviorDim]
	actOuts := G.Must(G.BroadcastHadamardProd(
		xUserBehaviors,
		G.Must(G.Sigmoid(
			//[batchSize, uBehaviorSize, 1]
			G.Must(G.Reshape(
				//[batchSize, uBehaviorSize]
				//	⊙
				//[:		, uBehaviorSize]
				G.Must(G.BroadcastHadamardProd(weight, din.att0, nil, []byte{0})),
				tensor.Shape{batchSize, uBehaviorSize, 1},
			)))),
		nil, []byte{2},
	))

	//actOuts := G.NewTensor(din.Graph(), model.DT, 2, G.WithShape(batchSize, uBehaviorDim), G.WithName("actOuts"), G.WithInit(G.Zeroes()))
	//for i := 0; i < uBehaviorSize; i++ {
	//	// xUserBehaviors[:, i, :], ub.shape: [batchSize, uBehaviorDim]
	//	ub := G.Must(G.Slice(xUserBehaviors, []tensor.Slice{nil, G.S(i)}...))
	//	// Concat all xUserBehaviors[i], outProducts, xItemFeature
	//	// actConcat.Shape() = [batchSize, uBehaviorDim+iFeatureDim+uBehaviorSize]
	//	actConcat := G.Must(G.Concat(1, ub, distance, xItemFeature))
	//	actOut := G.Must(G.BroadcastHadamardProd(
	//		ub,
	//		G.Must(G.Sigmoid(
	//			G.Must(G.Mul(
	//				G.Must(G.HadamardProd(actConcat, din.att0)),
	//				din.att1,
	//			)))), // [batchSize, 1]
	//		nil, []byte{1},
	//	)) // [batchSize, uBehaviorDim]
	//
	//	// Sum pooling
	//	actOuts = G.Must(G.Add(actOuts, actOut))
	//}
	actOutSum := G.Must(G.Mean(actOuts, 1))

	// Concat all xUserProfile, actOuts, xItemFeature, xCtxFeature
	concat := G.Must(G.Concat(1, xUserProfile, actOutSum, xItemFeature, xCtxFeature))

	// MLP

	// mlp0.Shape: [userProfileDim+userBehaviorDim+itemFeatureDim+contextFeatureDim, 200]
	// out.Shape: [batchSize, 200]
	mlp0Out := G.Must(G.Sigmoid(G.Must(G.Mul(concat, din.mlp0))))
	mlp0Out = G.Must(G.Dropout(mlp0Out, float64(din.d0)))
	// mlp1.Shape: [200, 80]
	// out.Shape: [batchSize, 80]
	mlp1Out := G.Must(G.Sigmoid(G.Must(G.Mul(mlp0Out, din.mlp1))))
	mlp1Out = G.Must(G.Dropout(mlp1Out, float64(din.d1)))
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
