package preprocessing

import (
	"fmt"
)

func ExampleBrentMinimizer() {
	f := func(x float64) float64 { return x * x }
	tol := 1e-8
	maxIter := 500
	fnMaxFev := func(nfev int) bool { return nfev > 1500 }
	bm := NewBrentMinimizer(f, tol, maxIter, fnMaxFev)
	bm.Brack = []float64{1, 2}
	x, fx, nIter, nFev := bm.Optimize()
	fmt.Printf("x: %.8g, fx: %.8g, nIter: %d, nFev: %d\n", x, fx, nIter, nFev)

	bm.Brack = []float64{-1, 0.5, 2}
	x, fx, nIter, nFev = bm.Optimize()
	fmt.Printf("x: %.8g, fx: %.8g, nIter: %d, nFev: %d\n", x, fx, nIter, nFev)
	// Output:
	// x: 0, fx: 0, nIter: 4, nFev: 9
	// x: -2.7755576e-17, fx: 7.7037198e-34, nIter: 5, nFev: 9

}
