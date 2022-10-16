package movielens

import (
	"github.com/auxten/edgeRec/nn/base"
	"gonum.org/v1/gonum/mat"
)

type dinImpl struct{}

func (d *dinImpl) Fit(X, Y mat.Matrix) base.Fiter {
	return d
}
