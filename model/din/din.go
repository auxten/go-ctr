package din

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	_ "net/http/pprof"

	rcmd "github.com/auxten/edgeRec/recommend"
	"github.com/auxten/edgeRec/utils"
	"github.com/pkg/errors"
	log "github.com/sirupsen/logrus"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	pb "gopkg.in/cheggaaa/pb.v1"
)

var dt = tensor.Float64

const (
	// magic numbers for din paper
	att0_1 = 36
	mlp0_1 = 200
	mlp1_2 = 80
)

type Model interface {
	learnable() G.Nodes
	Fwd(xUserProfile, ubMatrix, xItemFeature, xCtxFeature *G.Node, batchSize, uBehaviorSize, uBehaviorDim int) (err error)
	Out() *G.Node
	In() G.Nodes
	Graph() *G.ExprGraph
	Marshal() (data []byte, err error)
	Vm() G.VM
	SetVM(vm G.VM)
}

type DinNet struct {
	uProfileDim, uBehaviorSize, uBehaviorDim int
	iFeatureDim                              int
	cFeatureDim                              int

	g *G.ExprGraph

	vm G.VM

	//input nodes
	xUserProfile, xUbMatrix, xItemFeature, xCtxFeature *G.Node

	mlp0, mlp1, mlp2 *G.Node   // weights of MLP layers
	d0, d1           float64   // dropout probabilities
	att0, att1       []*G.Node // weights of Attention layers

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
	)

	// attention layer
	att0 := make([]*G.Node, m.UBehaviorSize)
	att1 := make([]*G.Node, m.UBehaviorSize)
	for i := 0; i < m.UBehaviorSize; i++ {
		att0[i] = G.NewMatrix(
			g,
			dt,
			G.WithShape(uBehaviorDim+iFeatureDim+uBehaviorSize*uBehaviorDim*iFeatureDim, att0_1),
			G.WithValue(m.Att0[i]),
			G.WithName(fmt.Sprintf("att0-%d", i)),
		)
		att1[i] = G.NewMatrix(
			g,
			dt,
			G.WithShape(att0_1, 1),
			G.WithValue(m.Att1[i]),
			G.WithName(fmt.Sprintf("att1-%d", i)),
		)
	}
	mlp0 := G.NewMatrix(g, dt, G.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+cFeatureDim, mlp0_1), G.WithName("mlp0"), G.WithValue(m.Mlp0))

	mlp1 := G.NewMatrix(g, dt, G.WithShape(mlp0_1, mlp1_2), G.WithName("mlp1"), G.WithValue(m.Mlp1))

	mlp2 := G.NewMatrix(g, dt, G.WithShape(mlp1_2, 1), G.WithName("mlp2"), G.WithValue(m.Mlp2))

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
		att0[i] = G.NewTensor(g, dt, 2, G.WithShape(uBehaviorDim+iFeatureDim+uBehaviorSize*uBehaviorDim*iFeatureDim, att0_1), G.WithName(fmt.Sprintf("att0-%d", i)), G.WithInit(G.Gaussian(0, 1)))
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

		d0: 0.01,
		d1: 0.01,

		mlp0: mlp0,
		mlp1: mlp1,
		mlp2: mlp2,
	}
}

//Fwd performs the forward pass
// xUserProfile: [batchSize, userProfileDim]
// xUserBehaviors: [batchSize, uBehaviorSize, uBehaviorDim]
// xItemFeature: [batchSize, iFeatureDim]
// xContextFeature: [batchSize, cFeatureDim]
func (din *DinNet) Fwd(xUserProfile, xUbMatrix, xItemFeature, xCtxFeature *G.Node, batchSize, uBehaviorSize, uBehaviorDim int) (err error) {
	iFeatureDim := xItemFeature.Shape()[1]
	if uBehaviorDim != iFeatureDim {
		return errors.Errorf("uBehaviorDim %d != iFeatureDim %d", uBehaviorDim, iFeatureDim)
	}
	xUserBehaviors := G.Must(G.Reshape(xUbMatrix, tensor.Shape{batchSize, uBehaviorSize, uBehaviorDim}))

	// outProduct should computed batch by batch!!!!
	outProdVecs := make([]*G.Node, batchSize)
	for i := 0; i < batchSize; i++ {
		// ubVec.Shape() = [uBehaviorSize * uBehaviorDim]
		ubVec := G.Must(G.Slice(xUbMatrix, G.S(i)))
		// item.Shape() = [iFeatureDim]
		itemVec := G.Must(G.Slice(xItemFeature, G.S(i)))
		// outProd.Shape() = [uBehaviorSize * uBehaviorDim, iFeatureDim]
		outProd := G.Must(G.OuterProd(ubVec, itemVec))
		outProdVecs[i] = G.Must(G.Reshape(outProd, tensor.Shape{uBehaviorSize * uBehaviorDim * iFeatureDim}))
	}
	//outProductsVec.Shape() = [batchSize * uBehaviorSize * uBehaviorDim * iFeatureDim]
	outProductsVec := G.Must(G.Concat(0, outProdVecs...))
	outProducts := G.Must(G.Reshape(outProductsVec, tensor.Shape{batchSize, uBehaviorSize * uBehaviorDim * iFeatureDim}))

	actOuts := G.NewTensor(din.Graph(), dt, 2, G.WithShape(batchSize, uBehaviorDim), G.WithName("actOuts"), G.WithInit(G.Zeroes()))
	for i := 0; i < uBehaviorSize; i++ {
		// xUserBehaviors[:, i, :], ub.shape: [batchSize, uBehaviorDim]
		ub := G.Must(G.Slice(xUserBehaviors, []tensor.Slice{nil, G.S(i)}...))
		// Concat all xUserBehaviors[i], outProducts, xItemFeature
		// actConcat.Shape() = [batchSize, uBehaviorDim+iFeatureDim+uBehaviorSize*uBehaviorDim*iFeatureDim]
		actConcat := G.Must(G.Concat(1, ub, outProducts, xItemFeature))
		actOut := G.Must(G.BroadcastHadamardProd(
			ub,
			G.Must(G.Rectify(
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
	mlp0Out := G.Must(G.LeakyRelu(G.Must(G.Mul(concat, din.mlp0)), 0.1))
	mlp0Out = G.Must(G.Dropout(mlp0Out, din.d0))
	// mlp1.Shape: [200, 80]
	// out.Shape: [batchSize, 80]
	mlp1Out := G.Must(G.LeakyRelu(G.Must(G.Mul(mlp0Out, din.mlp1)), 0.1))
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

func Train(uBehaviorSize, uBehaviorDim, uProfileDim, iFeatureDim, cFeatureDim int,
	numExamples, batchSize, epochs int,
	si *rcmd.SampleInfo,
	inputs, targets tensor.Tensor,
	//testInputs, testTargets tensor.Tensor,
	m Model,
) (err error) {
	rand.Seed(2120)

	g := m.Graph()
	xUserProfile := G.NewMatrix(g, dt, G.WithShape(batchSize, uProfileDim), G.WithName("xUserProfile"))
	//xUserBehaviors := G.NewTensor(g, dt, 3, G.WithShape(batchSize, uBehaviorSize, uBehaviorDim), G.WithName("xUserBehaviors"))
	xUserBehaviorMatrix := G.NewMatrix(g, dt, G.WithShape(batchSize, uBehaviorSize*uBehaviorDim), G.WithName("xUserBehaviorMatrix"))
	xItemFeature := G.NewMatrix(g, dt, G.WithShape(batchSize, iFeatureDim), G.WithName("xItemFeature"))
	xCtxFeature := G.NewMatrix(g, dt, G.WithShape(batchSize, cFeatureDim), G.WithName("xCtxFeature"))
	y := G.NewTensor(g, dt, 2, G.WithShape(batchSize, 1), G.WithName("y"))
	//m := NewDinNet(g, uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
	if err = m.Fwd(xUserProfile, xUserBehaviorMatrix, xItemFeature, xCtxFeature, batchSize, uBehaviorSize, uBehaviorDim); err != nil {
		log.Fatalf("%+v", err)
	}

	//losses := G.Must(G.HadamardProd(G.Must(G.Neg(G.Must(G.Log(m.out)))), y))
	//losses := G.Must(G.Square(G.Must(G.Sub(m.Out(), y))))
	positive := G.Must(G.HadamardProd(G.Must(G.Log(m.Out())), y))
	negative := G.Must(G.HadamardProd(G.Must(G.Log(G.Must(G.Sub(G.NewConstant(float64(1.0+1e-8)), m.Out())))), G.Must(G.Sub(G.NewConstant(float64(1.0)), y))))
	//negative := G.Must(G.Log(G.Must(G.Sub(G.NewConstant(float64(1.000000001)), m.Out()))))
	cost := G.Must(G.Neg(G.Must(G.Mean(G.Must(G.Add(positive, negative))))))

	// we want to track costs
	var costVal G.Value
	G.Read(cost, &costVal)

	var yOut G.Value
	G.Read(m.Out(), &yOut)

	if _, err = G.Grad(cost, m.learnable()...); err != nil {
		log.Fatal(err)
	}

	// debug
	//ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
	// log.Printf("%v", prog)
	// logger := log.New(os.Stderr, "", 0)
	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnable()...), gorgonia.WithLogger(logger), gorgonia.WithWatchlist())

	prog, locMap, err := G.Compile(g)
	if err != nil {
		log.Fatal(err)
	}
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g,
		G.WithPrecompiled(prog, locMap),
		G.BindDualValues(m.learnable()...),
		//G.TraceExec(),
		//G.WithInfWatch(),
		//G.WithNaNWatch(),
		//G.WithLogger(log.New(os.Stderr, "", 0)),
		//G.WithWatchlist(m.mlp2),
	)
	m.SetVM(vm)

	//solver := G.NewRMSPropSolver(G.WithBatchSize(float64(batchSize)))
	solver := G.NewAdamSolver(G.WithLearnRate(0.001))
	//defer func() {
	//	vm.Close()
	//	m.SetVM(nil)
	//}()
	// pprof
	// handlePprof(sigChan, doneChan)

	batches := numExamples / batchSize
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)

	for i := 0; i < epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * batchSize
			end := start + batchSize
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var (
				xUserProfileVal   tensor.Tensor
				xUserBehaviorsVal tensor.Tensor
				xItemFeatureVal   tensor.Tensor
				xCtxFeatureVal    tensor.Tensor
				yVal              tensor.Tensor
			)

			if xUserProfileVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.UserProfileRange[0], si.UserProfileRange[1])}...); err != nil {
				log.Fatalf("Unable to slice xUserProfileVal %v", err)
			}
			if err = G.Let(xUserProfile, xUserProfileVal); err != nil {
				log.Fatalf("Unable to let xUserProfileVal %v", err)
			}

			if xUserBehaviorsVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.UserBehaviorRange[0], si.UserBehaviorRange[1])}...); err != nil {
				log.Fatalf("Unable to slice xUserBehaviorsVal %v", err)
			}
			if err = G.Let(xUserBehaviorMatrix, xUserBehaviorsVal); err != nil {
				log.Fatalf("Unable to let xUserBehaviorsVal %v", err)
			}

			if xItemFeatureVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.ItemFeatureRange[0], si.ItemFeatureRange[1])}...); err != nil {
				log.Fatalf("Unable to slice xItemFeatureVal %v", err)
			}
			if err = G.Let(xItemFeature, xItemFeatureVal); err != nil {
				log.Fatalf("Unable to let xItemFeatureVal %v", err)
			}

			if xCtxFeatureVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.CtxFeatureRange[0], si.CtxFeatureRange[1])}...); err != nil {
				log.Fatalf("Unable to slice xCtxFeatureVal %v", err)
			}
			if err = G.Let(xCtxFeature, xCtxFeatureVal); err != nil {
				log.Fatalf("Unable to let xCtxFeatureVal %v", err)
			}

			if yVal, err = targets.Slice(G.S(start, end)); err != nil {
				log.Fatalf("Unable to slice y %v", err)
			}
			if err = G.Let(y, yVal); err != nil {
				log.Fatalf("Unable to let y %v", err)
			}

			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
			}
			if err = solver.Step(G.NodesToValueGrads(m.learnable())); err != nil {
				log.Fatalf("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, b, err)
			}
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)

		//log.Printf("Test accuracy %v | rocauc %v")
	}
	return
}

func InitForwardOnlyVm(uBehaviorSize, uBehaviorDim, uProfileDim, iFeatureDim, cFeatureDim int,
	batchSize int,
	m Model,
) (err error) {
	g := m.Graph()
	xUserProfile := G.NewMatrix(g, dt, G.WithShape(batchSize, uProfileDim), G.WithName("xUserProfile"))
	xUserBehaviorMatrix := G.NewMatrix(g, dt, G.WithShape(batchSize, uBehaviorSize*uBehaviorDim), G.WithName("xUserBehaviorMatrix"))
	xItemFeature := G.NewMatrix(g, dt, G.WithShape(batchSize, iFeatureDim), G.WithName("xItemFeature"))
	xCtxFeature := G.NewMatrix(g, dt, G.WithShape(batchSize, cFeatureDim), G.WithName("xCtxFeature"))
	if err = m.Fwd(xUserProfile, xUserBehaviorMatrix, xItemFeature, xCtxFeature,
		batchSize, uBehaviorSize, uBehaviorDim); err != nil {
		return
	}
	prog, locMap, err := G.Compile(g)
	if err != nil {
		return
	}
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g,
		G.WithPrecompiled(prog, locMap),
	)
	m.SetVM(vm)

	return
}

func Predict(m Model, numExamples, batchSize int, si *rcmd.SampleInfo, inputs tensor.Tensor) (y []float64, err error) {
	//input nodes
	inputNodes := m.In()
	xUserProfile := inputNodes[0]
	xUbMatrix := inputNodes[1]
	xItemFeature := inputNodes[2]
	xCtxFeature := inputNodes[3]

	//output node
	outputNode := m.Out()

	//vm
	vm := m.Vm()

	batches := numExamples / batchSize

	for b := 0; b < batches; b++ {
		start := b * batchSize
		end := start + batchSize
		if start >= numExamples {
			break
		}
		if end > numExamples {
			end = numExamples
		}

		var (
			xUserProfileVal   tensor.Tensor
			xUserBehaviorsVal tensor.Tensor
			xItemFeatureVal   tensor.Tensor
			xCtxFeatureVal    tensor.Tensor
		)

		if xUserProfileVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.UserProfileRange[0], si.UserProfileRange[1])}...); err != nil {
			log.Errorf("Unable to slice xUserProfileVal %v", err)
			return nil, err
		}
		if err = G.Let(xUserProfile, xUserProfileVal); err != nil {
			log.Errorf("Unable to let xUserProfileVal %v", err)
			return nil, err
		}

		if xUserBehaviorsVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.UserBehaviorRange[0], si.UserBehaviorRange[1])}...); err != nil {
			log.Errorf("Unable to slice xUserBehaviorsVal %v", err)
			return nil, err
		}
		if err = G.Let(xUbMatrix, xUserBehaviorsVal); err != nil {
			log.Errorf("Unable to let xUserBehaviorsVal %v", err)
			return nil, err
		}

		if xItemFeatureVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.ItemFeatureRange[0], si.ItemFeatureRange[1])}...); err != nil {
			log.Errorf("Unable to slice xItemFeatureVal %v", err)
			return nil, err
		}
		if err = G.Let(xItemFeature, xItemFeatureVal); err != nil {
			log.Errorf("Unable to let xItemFeatureVal %v", err)
			return nil, err
		}

		if xCtxFeatureVal, err = inputs.Slice([]tensor.Slice{G.S(start, end), G.S(si.CtxFeatureRange[0], si.CtxFeatureRange[1])}...); err != nil {
			log.Errorf("Unable to slice xCtxFeatureVal %v", err)
			return nil, err
		}
		if err = G.Let(xCtxFeature, xCtxFeatureVal); err != nil {
			log.Errorf("Unable to let xCtxFeatureVal %v", err)
			return nil, err
		}

		if err = vm.RunAll(); err != nil {
			log.Errorf("Failed at batch %d. Error: %v", b, err)
			return nil, err
		}
		vm.Reset()

		//get y
		yVal := outputNode.Value().Data().([]float64)
		y = append(y, yVal...)

	}
	return
}

func accuracy(prediction, y []float64) float64 {
	var ok float64
	for i := 0; i < len(prediction); i++ {
		if math.Round(prediction[i]-y[i]) == 0 {
			ok += 1.0
		}
	}
	return ok / float64(len(y))
}

func rocauc(pred, y []float64) float64 {
	boolY := make([]bool, len(y))
	for i := 0; i < len(y); i++ {
		boolY[i] = y[i] == 1.0
	}
	return utils.RocAuc(boolY, pred)
}
