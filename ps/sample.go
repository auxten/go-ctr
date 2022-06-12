package ps

import "math/rand"

// Sample is an input-target pair
type Sample struct {
	Input    []float64
	Response []float64
}

// Samples is a set of input-output pairs
type Samples []Sample

// Shuffle shuffles slice in-place
func (e Samples) Shuffle() {
	for i := range e {
		j := rand.Intn(i + 1)
		e[i], e[j] = e[j], e[i]
	}
}

// Split assigns each element to two new slices
// according to probability p
func (e Samples) Split(p float64) (first, second Samples) {
	for i := 0; i < len(e); i++ {
		if p > rand.Float64() {
			first = append(first, e[i])
		} else {
			second = append(second, e[i])
		}
	}
	return
}

// SplitSize splits slice into parts of size size
func (e Samples) SplitSize(size int) []Samples {
	res := make([]Samples, 0)
	for i := 0; i < len(e); i += size {
		res = append(res, e[i:min(i+size, len(e))])
	}
	return res
}

// SplitN splits slice into n parts
func (e Samples) SplitN(n int) []Samples {
	res := make([]Samples, n)
	for i, el := range e {
		res[i%n] = append(res[i%n], el)
	}
	return res
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}
