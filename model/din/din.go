package din

import (
	"fmt"
	"log"
	"math/rand"
	_ "net/http/pprof"
	"time"

	rcmd "github.com/auxten/edgeRec/recommend"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	pb "gopkg.in/cheggaaa/pb.v1"
)

var dt = tensor.Float64

type DinNet struct {
	uProfileDim, uBehaviorSize, uBehaviorDim int
	iFeatureDim                              int
	cFeatureDim                              int

	g                *G.ExprGraph
	mlp0, mlp1, mlp2 *G.Node   // weights of MLP layers
	att0, att1       []*G.Node // weights of Attention layers
	//d0, d1           float64 // dropout probabilities

	out *G.Node
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

func NewDinNet(g *G.ExprGraph,
	uProfileDim, uBehaviorSize, uBehaviorDim int,
	iFeatureDim int,
	ctxFeatureDim int,
) *DinNet {
	if uBehaviorDim != iFeatureDim {
		log.Fatalf("uBehaviorDim %d != iFeatureDim %d", uBehaviorDim, iFeatureDim)
	}
	// attention layer
	att0 := make([]*G.Node, uBehaviorSize)
	att1 := make([]*G.Node, uBehaviorSize)
	for i := 0; i < uBehaviorSize; i++ {
		att0[i] = G.NewTensor(g, dt, 2, G.WithShape(uBehaviorDim+iFeatureDim+uBehaviorSize*uBehaviorDim*iFeatureDim, 36), G.WithName("att0"), G.WithInit(G.GlorotN(1.0)))
		att1[i] = G.NewTensor(g, dt, 2, G.WithShape(36, 1), G.WithName("att1"), G.WithInit(G.GlorotN(1.0)))
	}

	// user behaviors are represented as a sequence of item embeddings. Before
	// being fed into the MLP, we need to flatten the sequence into a single with
	// sum pooling with Attention as the weights which is the key point of DIN model.
	mlp0 := G.NewMatrix(g, G.Float64, G.WithShape(uProfileDim+uBehaviorDim+iFeatureDim+ctxFeatureDim, 200), G.WithName("mlp0"), G.WithInit(G.GlorotN(1.0)))

	mlp1 := G.NewMatrix(g, G.Float64, G.WithShape(200, 80), G.WithName("mlp1"), G.WithInit(G.GlorotN(1.0)))

	mlp2 := G.NewMatrix(g, G.Float64, G.WithShape(80, 1), G.WithName("mlp2"), G.WithInit(G.GlorotN(1.0)))

	return &DinNet{
		uProfileDim:   uProfileDim,
		uBehaviorSize: uBehaviorSize,
		uBehaviorDim:  uBehaviorDim,
		iFeatureDim:   iFeatureDim,
		cFeatureDim:   ctxFeatureDim,

		g:    g,
		att0: att0,
		att1: att1,

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
func (d *DinNet) Fwd(xUserProfile, ubMatrix, xItemFeature, xCtxFeature *G.Node, batchSize, uBehaviorSize, uBehaviorDim int) (err error) {
	iFeatureDim := xItemFeature.Shape()[1]
	if uBehaviorDim != iFeatureDim {
		return errors.Errorf("uBehaviorDim %d != iFeatureDim %d", uBehaviorDim, iFeatureDim)
	}
	xUserBehaviors := G.Must(G.Reshape(ubMatrix, tensor.Shape{batchSize, uBehaviorSize, uBehaviorDim}))

	// outProduct should computed batch by batch!!!!
	outProdVecs := make([]*G.Node, batchSize)
	for i := 0; i < batchSize; i++ {
		// ubVec.Shape() = [uBehaviorSize * uBehaviorDim]
		ubVec := G.Must(G.Slice(ubMatrix, G.S(i)))
		// item.Shape() = [iFeatureDim]
		itemVec := G.Must(G.Slice(xItemFeature, G.S(i)))
		// outProd.Shape() = [uBehaviorSize * uBehaviorDim, iFeatureDim]
		outProd := G.Must(G.OuterProd(ubVec, itemVec))
		outProdVecs[i] = G.Must(G.Reshape(outProd, tensor.Shape{uBehaviorSize * uBehaviorDim * iFeatureDim}))
	}
	//outProductsVec.Shape() = [batchSize * uBehaviorSize * uBehaviorDim * iFeatureDim]
	outProductsVec := G.Must(G.Concat(0, outProdVecs...))
	outProducts := G.Must(G.Reshape(outProductsVec, tensor.Shape{batchSize, uBehaviorSize * uBehaviorDim * iFeatureDim}))

	actOuts := G.NewTensor(d.g, dt, 2, G.WithShape(batchSize, uBehaviorDim), G.WithName("actOuts"), G.WithInit(G.Zeroes()))
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
					G.Must(G.Mul(actConcat, d.att0[i])),
					d.att1[i],
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
	mlp0Out := G.Must(G.LeakyRelu(G.Must(G.Mul(concat, d.mlp0)), 0.1))
	// mlp1.Shape: [200, 80]
	// out.Shape: [batchSize, 80]
	mlp1Out := G.Must(G.LeakyRelu(G.Must(G.Mul(mlp0Out, d.mlp1)), 0.1))
	// mlp2.Shape: [80, 1]
	// out.Shape: [batchSize, 1]
	mlp2Out := G.Must(G.SoftMax(G.Must(G.Mul(mlp1Out, d.mlp2))))

	d.out = mlp2Out
	return
}

func Train(uBehaviorSize, uBehaviorDim, uProfileDim, iFeatureDim, cFeatureDim int,
	numExamples, batchSize, epochs int,
	si *rcmd.SampleInfo,
	inputs, targets tensor.Tensor,
) (err error) {
	rand.Seed(2120)

	g := G.NewGraph()
	xUserProfile := G.NewMatrix(g, dt, G.WithShape(batchSize, uProfileDim), G.WithName("xUserProfile"))
	//xUserBehaviors := G.NewTensor(g, dt, 3, G.WithShape(batchSize, uBehaviorSize, uBehaviorDim), G.WithName("xUserBehaviors"))
	xUserBehaviorMatrix := G.NewMatrix(g, dt, G.WithShape(batchSize, uBehaviorSize*uBehaviorDim), G.WithName("xUserBehaviorMatrix"))
	xItemFeature := G.NewMatrix(g, dt, G.WithShape(batchSize, iFeatureDim), G.WithName("xItemFeature"))
	xCtxFeature := G.NewMatrix(g, dt, G.WithShape(batchSize, cFeatureDim), G.WithName("xCtxFeature"))
	y := G.NewTensor(g, dt, 2, G.WithShape(batchSize, 1), G.WithName("y"))
	m := NewDinNet(g, uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
	if err = m.Fwd(xUserProfile, xUserBehaviorMatrix, xItemFeature, xCtxFeature, batchSize, uBehaviorSize, uBehaviorDim); err != nil {
		log.Fatalf("%+v", err)
	}

	losses := G.Must(G.HadamardProd(G.Must(G.Neg(G.Must(G.Log(m.out)))), y))
	cost := G.Must(G.Mean(losses))
	cost = G.Must(G.Neg(cost))

	// we want to track costs
	var costVal G.Value
	G.Read(cost, &costVal)

	if _, err = G.Grad(cost, m.learnable()...); err != nil {
		log.Fatal(err)
	}

	// debug
	// ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
	// log.Printf("%v", prog)
	// logger := log.New(os.Stderr, "", 0)
	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnable()...), gorgonia.WithLogger(logger), gorgonia.WithWatchlist())

	prog, locMap, err := G.Compile(g)
	if err != nil {
		log.Fatal(err)
	}
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(m.learnable()...))
	solver := G.NewRMSPropSolver(G.WithBatchSize(float64(batchSize)))
	defer vm.Close()
	// pprof
	// handlePprof(sigChan, doneChan)

	batches := numExamples / batchSize
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

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
	}
	return
}
