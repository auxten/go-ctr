package din

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestPRelu(t *testing.T) {
	Convey("prelu", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(4, 1), tensor.WithBacking([]float64{-1, -2, 3, 4})), G.WithName("x"))
		a := G.NewScalar(g, G.Float64, G.WithValue(0.1), G.WithName("a"))
		output := PRelu(x, a)
		//cost := G.Must(G.Mean(output))
		//
		//if _, err := G.Grad(cost, a); err != nil {
		//	log.Fatal(err)
		//}
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So(output.Value().Data(), ShouldResemble, []float64{-0.1, -0.2, 3, 4})
	})
}
