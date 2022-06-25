package feature

import (
	"reflect"
)

type numericalTransformer interface {
	Fit(vals []float64)
	Transform(val float64) float64
}

type stringTransformer interface {
	Fit(vals []string)
	Transform(val string) float64
}

type stringExpandingTransformer interface {
	Fit(vals []string)
	NumFeatures() int
	Transform(val string) []float64
}

// StructTransformer uses reflection to encode struct into feature vector.
// It uses struct tags to create feature transformers for each field.
// Since it is using reflection, there is a slight overhead for large structs, which can be seen in benchmarks.
// For better performance, use codegen version for your struct, refer to README of this repo.
type StructTransformer struct {
	Transformers []interface{}
}

// Fit will fit all field transformers
func (s *StructTransformer) Fit(_ []interface{}) {
	// TODO: go through encoders, make slice for each with data, call fit on that data
	panic("not implemented")
}

// Transform applies all field transformers
func (s *StructTransformer) Transform(v interface{}) []float64 {
	if v == nil || s == nil {
		return nil
	}

	if s.getNumFeatures() == 0 {
		return nil
	}

	features := make([]float64, 0, s.getNumFeatures())

	val := reflect.ValueOf(v)
	for i := 0; i < val.NumField() && i < len(s.Transformers); i++ {
		transformer := s.Transformers[i]
		if transformer == nil || reflect.ValueOf(transformer).IsNil() {
			continue
		}

		field := val.Field(i)
		switch field.Type().Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			features = append(features, s.transformNumerical(transformer, float64(field.Int()))...)
		case reflect.Float32, reflect.Float64:
			features = append(features, s.transformNumerical(transformer, field.Float())...)
		case reflect.String:
			features = append(features, s.transformString(transformer, field.String())...)
		default:
			panic("unsupported type in struct")
		}
	}

	return features
}

func (s *StructTransformer) getNumFeatures() int {
	count := 0
	for _, tr := range s.Transformers {
		if tr, ok := tr.(stringExpandingTransformer); ok {
			count += tr.NumFeatures()
		} else {
			count++
		}
	}
	return count
}

func (s *StructTransformer) transformNumerical(transformer interface{}, val float64) []float64 {
	if transformer, ok := transformer.(numericalTransformer); ok {
		return []float64{transformer.Transform(val)}
	}
	return nil
}

func (s *StructTransformer) transformString(transformer interface{}, val string) []float64 {
	if transformer, ok := transformer.(stringTransformer); ok {
		return []float64{transformer.Transform(val)}
	}
	if transformer, ok := transformer.(stringExpandingTransformer); ok {
		return transformer.Transform(val)
	}
	return nil
}
