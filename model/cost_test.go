package model

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestBCECostFuncs(t *testing.T) {
	Convey("Binary Cross Entropy", t, func() {
		g := G.NewGraph()

		yPred := G.NodeFromAny(g, tensor.New(tensor.WithShape(5, 1), tensor.WithBacking([]float32{0.4498, 0.9845, 0.4576, 0.3494, 0.2434})), G.WithName("yPred"))
		yTrue := G.NodeFromAny(g, tensor.New(tensor.WithShape(5, 1), tensor.WithBacking([]float32{0.2345, 0.5565, 0.3468, 0.1444, 0.3546})), G.WithName("yTrue"))
		output := BinaryCrossEntropy32(yPred, yTrue)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{})
		So(output.Value().Data(), ShouldAlmostEqual, 0.8746, 0.0001)
	})
	Convey("Binary Cross Entropy test against sklearn", t, func() {
		// Test case from Sklearn
		//  ```python
		//	from sklearn.metrics import log_loss
		//
		//	y = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
		//	y_pred = [0.19, 0.33, 0.47, 0.7, 0.74, 0.81, 0.86, 0.94, 0.97, 0.99]
		//	loss = log_loss(y, y_pred)
		//
		//	print('p(y) = {}'.format(np.round(y_pred, 2)))
		//	print('Log Loss / Cross Entropy = {:.4f}'.format(loss))
		//
		//	# Log Loss / Cross Entropy = 0.3335
		//  ```

		g := G.NewGraph()

		yPred := G.NodeFromAny(g, tensor.New(tensor.WithShape(10, 1),
			tensor.WithBacking([]float32{0.19, 0.33, 0.47, 0.7, 0.74, 0.81, 0.86, 0.94, 0.97, 0.99})), G.WithName("yPred"))
		yTrue := G.NodeFromAny(g, tensor.New(tensor.WithShape(10, 1),
			tensor.WithBacking([]float32{0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0})), G.WithName("yTrue"))
		output := BinaryCrossEntropy32(yPred, yTrue)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{})
		So(output.Value().Data(), ShouldAlmostEqual, 0.3335, 0.0001)
	})
}

func TestMSECostFuncs(t *testing.T) {
	Convey("Mean Squared Error", t, func() {
		g := G.NewGraph()

		yPred := G.NodeFromAny(g, tensor.New(tensor.WithShape(5, 1), tensor.WithBacking([]float32{0.1, 0.999999, 0.1, 0.9, 0.1})), G.WithName("yPred"))
		yTrue := G.NodeFromAny(g, tensor.New(tensor.WithShape(5, 1), tensor.WithBacking([]float32{0, 1, 0, 1, 0})), G.WithName("yTrue"))
		output := MSE32(yPred, yTrue)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{})
		So(output.Value().Data(), ShouldAlmostEqual, 0.008, 0.000001)
	})
}

func TestRMS32(t *testing.T) {
	Convey("Root Mean Squared Error", t, func() {
		g := G.NewGraph()

		yPred := G.NodeFromAny(g, tensor.New(tensor.WithShape(5, 1), tensor.WithBacking([]float32{0.1, 0.999999, 0.1, 0.9, 0.1})), G.WithName("yPred"))
		yTrue := G.NodeFromAny(g, tensor.New(tensor.WithShape(5, 1), tensor.WithBacking([]float32{0, 1, 0, 1, 0})), G.WithName("yTrue"))
		output := RMS32(yPred, yTrue)
		m := G.NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		defer m.Close()
		So([]int(output.Shape()), ShouldResemble, []int{})
		So(output.Value().Data(), ShouldAlmostEqual, 0.0894427, 0.000001)
	})
}
