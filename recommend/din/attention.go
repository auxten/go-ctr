package din

import "gonum.org/v1/gonum/mat"

func DinProduct(userItem *mat.Dense, itemFeature *mat.Dense) (ret *mat.Dense) {
	ret = &mat.Dense{}
	ret.MulElem(userItem, itemFeature)
	return
}
