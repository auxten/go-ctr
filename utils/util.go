package utils

import (
	"encoding/binary"
	"math"
)

func ConcatSlice(slices ...[]float64) []float64 {
	result := make([]float64, 0)
	for _, slice := range slices {
		result = append(result, slice...)
	}
	return result
}

func Float64toBytes(f float64) []byte {
	bits := math.Float64bits(f)
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, bits)
	return bytes
}
