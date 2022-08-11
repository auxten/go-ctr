package neuralnetwork

import "sort"

// IndexedXY implements sort.Slice to be used in Shuffle and sort.Sort
type IndexedXY struct {
	Idx, X, Y sort.Interface
}

func (s IndexedXY) Len() int           { return s.Idx.Len() }
func (s IndexedXY) Less(i, j int) bool { return s.Idx.Less(i, j) }
func (s IndexedXY) Swap(i, j int)      { s.Idx.Swap(i, j); s.X.Swap(i, j); s.Y.Swap(i, j) }
