package preprocessing

import (
	"math"
)

type bracketer struct {
	growLimit float64
	maxIter   int
}

// Bracket the minimum of the function.
// Given a function and distinct initial points, search in the
// downhill direction (as defined by the initital points) and return
// new points xa, xb, xc that bracket the minimum of the function
// f(xa) > f(xb) < f(xc). It doesn't always mean that obtained
// solution will satisfy xa<=x<=xb
func (b bracketer) bracket(f func(float64) float64, xa0, xb0 float64) (xa, xb, xc, fa, fb, fc float64, funcalls int) {
	var (
		tmp1, tmp2, val, denom, w, wlim, fw float64
		iter                                int
	)
	_gold := 1.618034 //# golden ratio: (1.0+sqrt(5.0))/2.0
	_verysmallNum := 1e-21
	xa, xb = xa0, xb0
	fa, fb = f(xa), f(xb)
	if fa < fb {
		xa, xb = xb, xa
		fa, fb = fb, fa
	}
	xc = xb + _gold*(xb-xa)
	fc = f(xc)
	funcalls = 3
	iter = 0
	for fc < fb {
		tmp1 = (xb - xa) * (fb - fc)
		tmp2 = (xb - xc) * (fb - fa)
		val = tmp2 - tmp1
		if math.Abs(val) < _verysmallNum {
			denom = 2.0 * _verysmallNum
		} else {
			denom = 2.0 * val
		}
		w = xb - ((xb-xc)*tmp2-(xb-xa)*tmp1)/denom
		wlim = xb + b.growLimit*(xc-xb)
		if iter > b.maxIter {
			panic("bracket: Too many iterations.")
		}
		iter++
		if (w-xc)*(xb-w) > 0.0 {
			fw = f(w)
			funcalls++
			if fw < fc {
				xa = xb
				xb = w
				fa = fb
				fb = fw
				return xa, xb, xc, fa, fb, fc, funcalls
			} else if fw > fb {
				xc = w
				fc = fw
				return xa, xb, xc, fa, fb, fc, funcalls
			}
			w = xc + _gold*(xc-xb)
			fw = f(w)
			funcalls++
		} else if (w-wlim)*(wlim-xc) >= 0.0 {
			w = wlim
			fw = f(w)
			funcalls++
		} else if (w-wlim)*(xc-w) > 0.0 {
			fw = f(w)
			funcalls++
			if fw < fc {
				xb = xc
				xc = w
				w = xc + _gold*(xc-xb)
				fb = fc
				fc = fw
				fw = f(w)
				funcalls++
			}
		} else {
			w = xc + _gold*(xc-xb)
			fw = f(w)
			funcalls++
		}
		xa = xb
		xb = xc
		xc = w
		fa = fb
		fb = fc
		fc = fw
	}
	return xa, xb, xc, fa, fb, fc, funcalls
}

// BrentMinimizer is the translation of class Brent in scipy/optimize/optimize.py
// Uses inverse parabolic interpolation when possible to speed up convergence of golden section method.
type BrentMinimizer struct {
	Func           func(float64) float64
	Tol            float64
	Maxiter        int
	mintol         float64
	cg             float64
	Xmin           float64
	Fval           float64
	Iter, Funcalls int
	Brack          []float64
	bracketer
	FnMaxFev func(int) bool
}

// NewBrentMinimizer returns an initialized *BrentMinimizer
func NewBrentMinimizer(fun func(float64) float64, tol float64, maxiter int, fnMaxFev func(int) bool) *BrentMinimizer {
	return &BrentMinimizer{
		Func:      fun,
		Tol:       tol,
		Maxiter:   maxiter,
		mintol:    1.0e-11,
		cg:        0.3819660,
		bracketer: bracketer{growLimit: 110, maxIter: 1000},
		FnMaxFev:  fnMaxFev,
	}
}

// SetBracket can be used to set initial bracket of BrentMinimizer. len(brack) must be between 1 and 3 inclusive.
func (bm *BrentMinimizer) SetBracket(brack []float64) {
	bm.Brack = make([]float64, len(brack))
	copy(bm.Brack, brack)
}
func (bm *BrentMinimizer) getBracketInfo() (float64, float64, float64, float64, float64, float64, int) {
	fun := bm.Func
	brack := bm.Brack
	var xa, xb, xc float64
	var fa, fb, fc float64
	var funcalls int
	switch len(brack) {
	case 0:
		xa, xb, xc, fa, fb, fc, funcalls = bm.bracketer.bracket(fun, 0, 1)
	case 2:
		xa, xb, xc, fa, fb, fc, funcalls = bm.bracketer.bracket(fun, brack[0], brack[1])
	case 3:
		xa, xb, xc = brack[0], brack[1], brack[2]
		if xa > xc {
			xa, xc = xc, xa
		}
		fa, fb, fc = fun(xa), fun(xb), fun(xc)
		if !((fb < fa) && (fb < fc)) {
			panic("getBracketInfo: not a brackeding interval")
		}
		funcalls = 3
	}
	return xa, xb, xc, fa, fb, fc, funcalls
}

// Optimize search the value of X minimizing bm.Func
func (bm *BrentMinimizer) Optimize() (x, fx float64, iter, funcalls int) {
	var xa, xb, xc, fb, _mintol, _cg, v, fv, w, fw, a, b, deltax, tol1, tol2, xmid, rat, tmp1, tmp2, p, dxTemp, u, fu float64
	if bm.FnMaxFev == nil {
		bm.FnMaxFev = func(int) bool { return false }
	}
	//# set up for optimization
	f := bm.Func

	xa, xb, xc, _, fb, _, funcalls = bm.getBracketInfo()
	_mintol = bm.mintol
	_cg = bm.cg
	// #################################
	// #BEGIN CORE ALGORITHM
	//#################################
	//x = w = v = xb
	v, w, x = xb, xb, xb
	//fw = fv = fx = func(*((x,) + self.args))
	fx = fb
	fv, fw = fx, fx
	if xa < xc {
		a = xa
		b = xc
	} else {
		a = xc
		b = xa
	}
	deltax = 0.0
	funcalls++
	iter = 0
	for iter < bm.Maxiter && !bm.FnMaxFev(funcalls) {
		tol1 = bm.Tol*math.Abs(x) + _mintol
		tol2 = 2.0 * tol1
		xmid = 0.5 * (a + b)
		//# check for convergence
		if math.Abs(x-xmid) < (tol2 - 0.5*(b-a)) {
			break
		}
		// # XXX In the first iteration, rat is only bound in the true case
		// # of this conditional. This used to cause an UnboundLocalError
		// # (gh-4140). It should be set before the if (but to what?).
		if math.Abs(deltax) <= tol1 {
			if x >= xmid {
				deltax = a - x //# do a golden section step
			} else {
				deltax = b - x
			}
			rat = _cg * deltax
		} else { //# do a parabolic step
			tmp1 = (x - w) * (fx - fv)
			tmp2 = (x - v) * (fx - fw)
			p = (x-v)*tmp2 - (x-w)*tmp1
			tmp2 = 2.0 * (tmp2 - tmp1)
			if tmp2 > 0.0 {
				p = -p
			}
			tmp2 = math.Abs(tmp2)
			dxTemp = deltax
			deltax = rat
			//# check parabolic fit
			if (p > tmp2*(a-x)) && (p < tmp2*(b-x)) &&
				(math.Abs(p) < math.Abs(0.5*tmp2*dxTemp)) {
				rat = p * 1.0 / tmp2 //# if parabolic step is useful.
				u = x + rat
				if (u-a) < tol2 || (b-u) < tol2 {
					if xmid-x >= 0 {
						rat = tol1
					} else {
						rat = -tol1
					}
				}
			} else {
				if x >= xmid {
					deltax = a - x //# if it's not do a golden section step
				} else {
					deltax = b - x
				}
				rat = _cg * deltax
			}
		}
		if math.Abs(rat) < tol1 { //# update by at least tol1
			if rat >= 0 {
				u = x + tol1
			} else {
				u = x - tol1
			}
		} else {
			u = x + rat
		}
		fu = f(u) //# calculate new output value
		funcalls++

		if fu > fx { //# if it's bigger than current
			if u < x {
				a = u
			} else {
				b = u
			}
			if (fu <= fw) || (w == x) {
				v = w
				w = u
				fv = fw
				fw = fu
			} else if (fu <= fv) || (v == x) || (v == w) {
				v = u
				fv = fu
			}
		} else {
			if u >= x {
				a = x
			} else {
				b = x
			}
			v = w
			w = x
			x = u
			fv = fw
			fw = fx
			fx = fu
		}
		iter++
	}
	// #################################
	// #END CORE ALGORITHM
	// #################################
	bm.Xmin, bm.Fval, bm.Iter, bm.Funcalls = x, fx, iter, funcalls
	return
}
