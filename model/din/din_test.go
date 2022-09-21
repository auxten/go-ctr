package din

import (
	"testing"

	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
	G "gorgonia.org/gorgonia"
)

func TestDin(t *testing.T) {
	Convey("Din simple Grad", t, func() {
		var (
			err           error
			batchSize     = 2
			uBehaviorSize = 3
			uBehaviorDim  = 2
			uProfileDim   = 2
			iFeatureDim   = 2
			cFeatureDim   = 2
			numExamples   = 2
			epochs        = 2
		)

		g := G.NewGraph()
		xUserProfile := G.NewMatrix(g, G.Float64, G.WithShape(batchSize, uProfileDim), G.WithInit(G.RangedFrom(1)), G.WithName("xUserProfile"))
		xUserBehavior := G.NewTensor(g, G.Float64, 3, G.WithShape(batchSize, uBehaviorSize, uBehaviorDim), G.WithInit(G.RangedFrom(10)), G.WithName("xUserBehavior"))
		xItemFeature := G.NewMatrix(g, G.Float64, G.WithShape(batchSize, iFeatureDim), G.WithInit(G.RangedFrom(100)), G.WithName("xItemFeature"))
		xContextFeature := G.NewMatrix(g, G.Float64, G.WithShape(batchSize, cFeatureDim), G.WithInit(G.RangedFrom(1000)), G.WithName("xContextFeature"))

		din := NewDinNet(g, uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
		if err = din.Fwd(xUserProfile, xUserBehavior, xItemFeature, xContextFeature); err != nil {
			log.Fatalf("%+v", err)
		}

		y := G.NewMatrix(g, G.Float64, G.WithShape(batchSize, 1), G.WithInit(G.RangedFrom(1)), G.WithName("y"))
		losses := G.Must(G.HadamardProd(G.Must(G.Neg(G.Must(G.Log(din.out)))), y))
		cost := G.Must(G.Mean(losses))
		cost = G.Must(G.Neg(cost))

		// we wanna track costs
		var costVal G.Value
		G.Read(cost, &costVal)

		if _, err = G.Grad(cost, din.learnables()...); err != nil {
			log.Fatal(err)
		}

	})
}
