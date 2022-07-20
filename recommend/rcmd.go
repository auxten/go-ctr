package recommend

import (
	"github.com/auxten/edgeRec/nn/base"
	nn "github.com/auxten/edgeRec/nn/neural_network"
	"github.com/auxten/edgeRec/ps"
	"github.com/auxten/edgeRec/utils"
	log "github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"
)

type Tensor []float64

type RecSys interface {
	UserFeaturer
	ItemFeaturer
	Trainer
}

type Predictor interface {
	UserFeaturer
	ItemFeaturer
	base.Predicter
}

type Trainer interface {
	SampleGenerator() (<-chan Sample, error)
}

type UserFeaturer interface {
	GetUserFeature(int) Tensor
}

type ItemFeaturer interface {
	GetItemFeature(int) Tensor
}

type PreRanker interface {
	PreRank() error
}

type PreTrainer interface {
	ItemSequencer
	PreTrain() error
}

//ItemSequencer is an interface to generate user generated time-series item sequence
//for training the item2vec embedding model
type ItemSequencer interface {
	ItemSeqGenerator() (<-chan string, error)
}

type ItemScore struct {
	ItemId int     `json:"itemId"`
	Score  float64 `json:"score"`
}

type Sample struct {
	UserId int     `json:"userId"`
	ItemId int     `json:"itemId"`
	Label  float64 `json:"label"`
}

func Train(recSys RecSys) (model Predictor, err error) {
	if preTrain, ok := recSys.(PreTrainer); ok {
		err = preTrain.PreTrain()
		if err != nil {
			return
		}
	}
	trainSample := GetSample(recSys)
	sampleLen := len(trainSample)
	sampleDense := mat.NewDense(sampleLen, len(trainSample[0].Input), nil)
	for i, sample := range trainSample {
		sampleDense.SetRow(i, sample.Input)
	}
	yClass := mat.NewDense(sampleLen, 1, nil)
	for i, sample := range trainSample {
		yClass.Set(i, 0, sample.Response[0])
	}
	mlp := nn.NewMLPClassifier(
		[]int{len(trainSample[0].Input), len(trainSample[0].Input)},
		"logistic", "adam", 0.,
	)
	mlp.Shuffle = true
	mlp.Verbose = true
	mlp.RandomState = base.NewLockedSource(1)
	mlp.BatchSize = 10
	mlp.MaxIter = 10
	mlp.LearningRate = "adaptive"
	mlp.LearningRateInit = .003
	mlp.NIterNoChange = 20

	//start training
	log.Infof("\nstart training with %d samples\n", sampleLen)
	mlp.Fit(sampleDense, yClass)
	type modelImpl struct {
		UserFeaturer
		ItemFeaturer
		base.Predicter
	}
	model = &modelImpl{
		UserFeaturer: recSys,
		ItemFeaturer: recSys,
		Predicter:    mlp,
	}

	return
}

func Rank(recSys Predictor, userId int, itemIds []int) (itemScores []ItemScore, err error) {
	if preRanker, ok := recSys.(PreRanker); ok {
		err = preRanker.PreRank()
		if err != nil {
			return
		}
	}
	itemScores = make([]ItemScore, len(itemIds))
	userFeature := recSys.GetUserFeature(userId)
	for i, itemId := range itemIds {
		itemFeature := recSys.GetItemFeature(itemId)
		xSlice := utils.ConcatSlice(userFeature, itemFeature)
		x := mat.NewDense(1, len(xSlice), xSlice)
		y := mat.NewDense(1, 1, nil)

		score := recSys.Predict(x, y)
		itemScores[i] = ItemScore{itemId, score.At(0, 0)}
	}
	return
}

func GetSample(recSys RecSys) (sample ps.Samples) {
	sampleGen, ok := recSys.(Trainer)
	if !ok {
		panic("sample generator not implemented")
	}
	sampleCh, err := sampleGen.SampleGenerator()
	if err != nil {
		panic(err)
	}
	var (
		userFeatureWidth, itemFeatureWidth int
	)

	for s := range sampleCh {
		userFeature := recSys.GetUserFeature(s.UserId)
		itemFeature := recSys.GetItemFeature(s.ItemId)
		if userFeatureWidth == 0 {
			userFeatureWidth = len(userFeature)
		}
		if len(userFeature) != userFeatureWidth {
			log.Errorf("user feature length mismatch: %v:%v",
				userFeatureWidth, len(userFeature))
			continue
		}
		if itemFeatureWidth == 0 {
			itemFeatureWidth = len(itemFeature)
		}
		if len(itemFeature) != itemFeatureWidth {
			log.Errorf("item feature length mismatch: %v:%v",
				itemFeatureWidth, len(itemFeature))
			continue
		}

		sample = append(sample, ps.Sample{Input: utils.ConcatSlice(userFeature, itemFeature), Response: Tensor{s.Label}})
		if len(sample)%100 == 0 {
			log.Infof("sample size: %d", len(sample))
		}
	}

	return
}
