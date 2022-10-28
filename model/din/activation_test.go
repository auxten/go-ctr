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

func TestEucDistance(t *testing.T) {
	Convey("euc distance 2 dim", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float64{1, 2, -3, 4, -1, 0, -1, 2})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float64{1, 2, -3, 4, 0, 1, 0, 1})), G.WithName("y"))
		output := EucDistance(x, y)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{2})
		So(output.Value().Data(), ShouldResemble, []float64{0, 2})
	})
	Convey("euc distance 3 dim no broadcast", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 2, 2), tensor.WithBacking([]float64{1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 2, 2), tensor.WithBacking([]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})), G.WithName("y"))
		output := EucDistance(x, y)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{3, 2})
		So(output.Value().Data(), ShouldResemble, []float64{1, 0, 1, 1, 0, 1})
	})
	Convey("euc distance 3 dim broadcast", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 2, 2), tensor.WithBacking([]float64{0, 2, 0, 0, -1, 1, -1, 1, 1, 2, 2, 2})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 1, 2), tensor.WithBacking([]float64{0, 1, 0, 1, 1, 2})), G.WithName("y"))
		output := EucDistance(x, y)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{3, 2})
		So(output.Value().Data(), ShouldResemble, []float64{1, 1, 1, 1, 0, 1})
	})
}
