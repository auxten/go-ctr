package model

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestPRelu(t *testing.T) {
	Convey("prelu", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(4, 1), tensor.WithBacking([]float32{-1, -2, 3, 4})), G.WithName("x"))
		a := G.NewScalar(g, G.Float32, G.WithValue(float32(0.1)), G.WithName("a"))
		output := PRelu32(x, a)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So(output.Value().Data(), ShouldResemble, []float32{-0.1, -0.2, 3, 4})
	})
}

func TestEucDistance(t *testing.T) {
	Convey("euc distance 2 dim", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{
			1, 2, -3, 4,
			-1, 0, -1, 2,
		})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{
			1, 2, -3, 4,
			0, 1, 0, 1,
		})), G.WithName("y"))
		output, err := EucDistance(x, y)
		So(err, ShouldBeNil)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{2})
		So(output.Value().Data(), ShouldResemble, []float32{0, 2})
	})
	Convey("euc distance 3 dim no broadcast", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 2, 2), tensor.WithBacking([]float32{
			1, 0,
			0, 0,
			1, 0,
			1, 0,
			0, 0,
			1, 0,
		})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 2, 2), tensor.WithBacking([]float32{
			0, 0,
			0, 0,
			0, 0,
			0, 0,
			0, 0,
			0, 0,
		})), G.WithName("y"))
		output, err := EucDistance(x, y)
		So(err, ShouldBeNil)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{3, 2})
		So(output.Value().Data(), ShouldResemble, []float32{1, 0, 1, 1, 0, 1})
	})
	Convey("euc distance 3 dim broadcast y", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 2, 2), tensor.WithBacking([]float32{
			0, 2,
			0, 0,
			-1, 1,
			-1, 1,
			1, 2,
			2, 2,
		})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 1, 2), tensor.WithBacking([]float32{
			0, 1,
			//0, 1, // broadcast expected
			0, 1,
			//0, 1, // broadcast expected
			1, 2,
			//1, 2, // broadcast expected
		})), G.WithName("y"))
		output, err := EucDistance(x, y)
		So(err, ShouldBeNil)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{3, 2})
		So(output.Value().Data(), ShouldResemble, []float32{1, 1, 1, 1, 0, 1})
	})
	Convey("euc distance 3 dim broadcast x", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 1, 2), tensor.WithBacking([]float32{
			0, 1,
			//0, 1, // broadcast expected
			0, 1,
			//0, 1, // broadcast expected
			1, 2,
			//1, 2, // broadcast expected
		})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(3, 2, 2), tensor.WithBacking([]float32{
			0, 2,
			0, 0,
			-1, 1,
			-1, 1,
			1, 2,
			2, 2,
		})), G.WithName("y"))
		output, err := EucDistance(x, y)
		So(err, ShouldBeNil)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{3, 2})
		So(output.Value().Data(), ShouldResemble, []float32{1, 1, 1, 1, 0, 1})
	})
	Convey("dim not match", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 6), tensor.WithBacking(
			[]float32{
				1, 1, 1, 1, 0, 0,
				1, 1, 0, 0, 1, 1,
			})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 2, 6), tensor.WithBacking(
			[]float32{
				1, 1, 0, 0, 1, 1,
				1, 1, 0, 0, 1, 1,
				1, 1, 1, 1, 0, 0,
				1, 1, 1, 1, 0, 0,
			})), G.WithName("y"))
		_, err := EucDistance(x, y)
		So(err, ShouldNotBeNil)
	})
}

func TestCosineDistance(t *testing.T) {
	Convey("cosine distance 2 dim", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 6), tensor.WithBacking([]float32{
			1, 1, 1, 1, 0, 0,
			1, 1, 0, 0, 1, 1,
		})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 6), tensor.WithBacking([]float32{
			1, 1, 0, 0, 1, 1,
			1, 1, 1, 1, 0, 0,
		})), G.WithName("y"))
		output, err := CosineDistance(x, y)
		So(err, ShouldBeNil)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{2})
		So(output.Value().Data(), ShouldResemble, []float32{0.5, 0.5})
	})
	Convey("cosine distance 3 dim", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 1, 6), tensor.WithBacking([]float32{
			1, 1, 1, 1, 0, 0,
			1, 1, 0, 0, 1, 1,
		})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 1, 6), tensor.WithBacking([]float32{
			1, 1, 0, 0, 1, 1,
			1, 1, 1, 1, 0, 0,
		})), G.WithName("y"))
		output, err := CosineDistance(x, y)
		So(err, ShouldBeNil)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{2, 1})
		So(output.Value().Data(), ShouldResemble, []float32{0.5, 0.5})
	})
	Convey("cosine distance 3 dim broadcast", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 1, 6), tensor.WithBacking(
			[]float32{
				1, 1, 1, 1, 0, 0,
				//1, 1, 1, 1, 0, 0, // broadcast expected
				1, 1, 0, 0, 1, 1,
				//1, 1, 0, 0, 1, 1, // broadcast expected
			})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 2, 6), tensor.WithBacking(
			[]float32{
				1, 1, 0, 0, 1, 1,
				1, 1, 0, 0, 1, 1,
				1, 1, 1, 1, 0, 0,
				1, 1, 1, 1, 0, 0,
			})), G.WithName("y"))
		output, err := CosineDistance(x, y)
		So(err, ShouldBeNil)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{2, 2})
		So(output.Value().Data(), ShouldResemble, []float32{0.5, 0.5, 0.5, 0.5})
	})
	Convey("dim not match", t, func() {
		g := G.NewGraph()
		x := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 6), tensor.WithBacking(
			[]float32{
				1, 1, 1, 1, 0, 0,
				1, 1, 0, 0, 1, 1,
			})), G.WithName("x"))
		y := G.NodeFromAny(g, tensor.New(tensor.WithShape(2, 2, 6), tensor.WithBacking(
			[]float32{
				1, 1, 0, 0, 1, 1,
				1, 1, 0, 0, 1, 1,
				1, 1, 1, 1, 0, 0,
				1, 1, 1, 1, 0, 0,
			})), G.WithName("y"))
		_, err := CosineDistance(x, y)
		So(err, ShouldNotBeNil)
	})
}
