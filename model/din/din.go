package din

import (
	"fmt"
	"log"
	"math/rand"
	_ "net/http/pprof"
	"time"

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

func (din *DinNet) learnables() G.Nodes {
	ret := make(G.Nodes, 3, 3+2*din.uBehaviorSize)
	ret[0] = din.mlp0
	ret[1] = din.mlp1
	ret[2] = din.mlp2
	ret = append(ret, din.att0...)
	ret = append(ret, din.att1...)
	return ret
}

func NewDinNet(g *G.ExprGraph,
	userProfileDim, userBehaviorSize, userBehaviorDim int,
	itemFeatureDim int,
	contextFeatureDim int,
) *DinNet {
	if userBehaviorDim != itemFeatureDim {
		log.Fatalf("userBehaviorDim %d != itemFeatureDim %d", userBehaviorDim, itemFeatureDim)
	}
	// attention layer
	att0 := make([]*G.Node, userBehaviorSize)
	att1 := make([]*G.Node, userBehaviorSize)
	for i := 0; i < userBehaviorSize; i++ {
		att0[i] = G.NewTensor(g, dt, 2, G.WithShape(userBehaviorDim*3, 36), G.WithName("att0"), G.WithInit(G.GlorotN(1.0)))
		att1[i] = G.NewTensor(g, dt, 2, G.WithShape(36, 1), G.WithName("att1"), G.WithInit(G.GlorotN(1.0)))
	}

	// user behaviors are represented as a sequence of item embeddings. Before
	// being fed into the MLP, we need to flatten the sequence into a single with
	// sum pooling with Attention as the weights which is the key point of DIN model.
	mlp0 := G.NewMatrix(g, G.Float64, G.WithShape(userProfileDim+userBehaviorDim+itemFeatureDim+contextFeatureDim, 200), G.WithName("mlp0"), G.WithInit(G.GlorotN(1.0)))

	mlp1 := G.NewMatrix(g, G.Float64, G.WithShape(200, 80), G.WithName("mlp1"), G.WithInit(G.GlorotN(1.0)))

	mlp2 := G.NewMatrix(g, G.Float64, G.WithShape(80, 1), G.WithName("mlp2"), G.WithInit(G.GlorotN(1.0)))

	return &DinNet{
		uProfileDim:   userProfileDim,
		uBehaviorSize: userBehaviorSize,
		uBehaviorDim:  userBehaviorDim,
		iFeatureDim:   itemFeatureDim,
		cFeatureDim:   contextFeatureDim,

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
func (d *DinNet) Fwd(xUserProfile, xUserBehaviors, xItemFeature, xCtxFeature *G.Node) (err error) {
	batchsize, uBehaviorSize, uBehaviorDim := xUserBehaviors.Shape()[0], xUserBehaviors.Shape()[1], xUserBehaviors.Shape()[2]
	iFeatureDim := xItemFeature.Shape()[1]
	if uBehaviorDim != iFeatureDim {
		return errors.Errorf("uBehaviorDim %d != xItemFeature.Shape()[1] %d", uBehaviorDim, iFeatureDim)
	}
	ub2d := G.Must(G.Reshape(xUserBehaviors, tensor.Shape{batchsize, uBehaviorSize * uBehaviorDim}))

	// outProduct should computed batch by batch!!!!
	outProdVecs := make([]*G.Node, batchsize)
	for i := 0; i < batchsize; i++ {
		// ubVec.Shape() = [uBehaviorSize * uBehaviorDim]
		ubVec := G.Must(G.Slice(ub2d, G.S(i)))
		// item.Shape() = [iFeatureDim]
		itemVec := G.Must(G.Slice(xItemFeature, G.S(i)))
		// outProd.Shape() = [uBehaviorSize * uBehaviorDim, iFeatureDim]
		outProd := G.Must(G.OuterProd(ubVec, itemVec))
		outProdVecs[i] = G.Must(G.Reshape(outProd, tensor.Shape{uBehaviorSize * uBehaviorDim * iFeatureDim}))
	}
	//outProducts.Shape() = [batchSize, uBehaviorSize * uBehaviorDim * iFeatureDim]
	outProducts := G.Must(G.Concat(0, outProdVecs...))

	actOuts := G.NewTensor(d.g, dt, 2, G.WithShape(batchsize, uBehaviorDim), G.WithName("actOuts"))
	for i := 0; i < uBehaviorSize; i++ {
		// xUserBehaviors[:, i, :], ub.shape: [batchSize, uBehaviorDim]
		ub := G.Must(G.Slice(xUserBehaviors, []tensor.Slice{nil, G.S(i)}...))
		// Concat all xUserBehaviors[i], outProducts, xItemFeature
		actConcat := G.Must(G.Concat(1, ub, outProducts, xItemFeature))
		actOut := G.Must(G.Mul(
			ub,
			G.Must(G.Rectify(
				G.Must(G.Mul(
					G.Must(G.Mul(actConcat, d.att0[i])),
					d.att1[i],
				)))),
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

func Train(uBehaviorSize, uBehaviorDim, uProfileDim, iFeatureDim, cFeatureDim, numExamples, batchSize, epochs int,

) {

	var (
		inputs, targets              tensor.Tensor
		err                          error
		xUserProfile, xCtxFeature    *G.Node
		xUserBehaviors, xItemFeature *G.Node
	)
	rand.Seed(1337)

	//bs := xUserProfile.Shape()[0]
	// := xUserBehaviors.Shape()[1], xUserBehaviors.Shape()[2]
	//uProfileDim := xUserProfile.Shape()[1]
	//iFeatureDim := xItemFeature.Shape()[1]
	//cFeatureDim := xCtxFeature.Shape()[1]

	//numExamples := inputs.Shape()[0]
	//
	//if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
	//	log.Fatal(err)
	//}
	g := G.NewGraph()
	//x := G.NewTensor(g, dt, 4, G.WithShape(bs, 1, 28, 28), G.WithName("x"))
	//y := G.NewMatrix(g, dt, G.WithShape(bs, 10), G.WithName("y"))
	m := NewDinNet(g, uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
	if err = m.Fwd(xUserProfile, xUserBehaviors, xItemFeature, xCtxFeature); err != nil {
		log.Fatalf("%+v", err)
	}

	losses := G.Must(G.HadamardProd(G.Must(G.Neg(G.Must(G.Log(m.out)))), y))
	cost := G.Must(G.Mean(losses))
	cost = G.Must(G.Neg(cost))

	// we wanna track costs
	var costVal G.Value
	G.Read(cost, &costVal)

	if _, err = G.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// debug
	// ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
	// log.Printf("%v", prog)
	// logger := log.New(os.Stderr, "", 0)
	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...), gorgonia.WithLogger(logger), gorgonia.WithWatchlist())

	prog, locMap, _ := G.Compile(g)
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(m.learnables()...))
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

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(G.S(start, end)); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(G.S(start, end)); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(batchSize, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			G.Let(x, xVal)
			G.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
			}
			if err = solver.Step(G.NodesToValueGrads(m.learnables())); err != nil {
				log.Fatalf("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, b, err)
			}
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)

	}
}
