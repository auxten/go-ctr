package feature

// todo: do normalization with database
import (
	"hash/fnv"
	"strings"
)

func SimpleOneHot(value int, size int) []float64 {
	result := make([]float64, size)
	result[value] = 1
	return result
}

func HashOneHot(buf []byte, size int) []float64 {
	result := make([]float64, size)
	hash := fnv.New32()
	_, err := hash.Write(buf)
	if err != nil {
		return nil
	}
	result[int(hash.Sum32())%size] = 1
	return result
}

func StringSplitMultiHot(str string, sep string, size int) []float64 {
	result := make([]float64, size)
	for _, s := range strings.Split(str, sep) {
		hash := fnv.New32()
		hash.Write([]byte(strings.ToLower(s)))
		result[int(hash.Sum32())%size] = 1
	}
	return result
}
