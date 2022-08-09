package din

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestActivation(t *testing.T) {
	Convey("product", t, func() {
		userItem := mat.NewDense(1, 4, []float64{
			1, 2, 3, 4,
		})
		itemFeature := mat.NewDense(1, 4, []float64{
			1, 2, 3, 4,
		})
		c := DinProduct(userItem, itemFeature)
		So(c, ShouldResemble, mat.NewDense(1, 4, []float64{
			1, 4, 9, 16,
		}))
		fc := mat.Formatted(c, mat.Prefix("    "), mat.Squeeze())
		fmt.Printf("c = %v", fc)
	})
}
