package recommend

import (
	"math/rand"
	"strconv"
	"time"

	"github.com/auxten/edgeRec/feature/embedding"
	"github.com/auxten/edgeRec/feature/embedding/model"
	"github.com/auxten/edgeRec/feature/embedding/model/word2vec"
	"github.com/auxten/edgeRec/nn/base"
	"github.com/auxten/edgeRec/ps"
	"github.com/auxten/edgeRec/utils"
	"github.com/karlseguin/ccache/v2"
	log "github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"
)

const (
	ItemEmbDim           = 10
	ItemEmbWindow        = 5
	userFeatureCacheSize = 200000
	itemFeatureCacheSize = 2000000
)

var (
	itemEmbeddingModel model.Model
	itemEmbeddingMap   word2vec.EmbeddingMap
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
	GetUserFeature(int) (Tensor, error)
}

type ItemFeaturer interface {
	GetItemFeature(int) (Tensor, error)
}

type PreRanker interface {
	PreRank() error
}

type PreTrainer interface {
	PreTrain() error
}

// ItemEmbedding is an interface used to generate item embedding with item2vec model
// by just providing a behavior based item sequence.
// Example: user liked items sequence, user bought items sequence, user viewed items sequence
type ItemEmbedding interface {
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

func Train(recSys RecSys, mlp base.Fiter) (model Predictor, err error) {
	rand.Seed(0)

	if preTrain, ok := recSys.(PreTrainer); ok {
		err = preTrain.PreTrain()
		if err != nil {
			log.Errorf("pre train error: %v", err)
			return
		}
	}

	if itemEbd, ok := recSys.(ItemEmbedding); ok {
		itemEmbeddingModel, err = GetItemEmbeddingModelFromUb(itemEbd)
		if err != nil {
			log.Errorf("get item embedding model error: %v", err)
			return
		}
		itemEmbeddingMap, err = itemEmbeddingModel.GenEmbeddingMap()
		if err != nil {
			log.Errorf("get item embedding map error: %v", err)
			return
		}
	}

	trainSample, err := GetSample(recSys)
	sampleLen := len(trainSample)
	sampleDense := mat.NewDense(sampleLen, len(trainSample[0].Input), nil)
	for i, sample := range trainSample {
		sampleDense.SetRow(i, sample.Input)
	}
	yClass := mat.NewDense(sampleLen, 1, nil)
	for i, sample := range trainSample {
		yClass.Set(i, 0, sample.Response[0])
	}

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
		Predicter:    mlp.(base.Predicter),
	}

	return
}

func Rank(recSys Predictor, userId int, itemIds []int) (itemScores []ItemScore, err error) {
	if preRanker, ok := recSys.(PreRanker); ok {
		err = preRanker.PreRank()
		if err != nil {
			log.Errorf("pre rank error: %v", err)
			return
		}
	}
	var (
		userFeature, itemFeature Tensor
	)
	itemScores = make([]ItemScore, len(itemIds))
	userFeature, err = recSys.GetUserFeature(userId)
	if err != nil {
		log.Errorf("get user feature error: %v", err)
		return
	}
	for i, itemId := range itemIds {
		itemFeature, err = recSys.GetItemFeature(itemId)
		if err != nil {
			log.Infof("get item feature failed: %v", err)
			return
		}
		xSlice := utils.ConcatSlice(userFeature, itemFeature)
		x := mat.NewDense(1, len(xSlice), xSlice)
		y := mat.NewDense(1, 1, nil)

		score := recSys.Predict(x, y)
		itemScores[i] = ItemScore{itemId, score.At(0, 0)}
	}
	return
}

func BatchPredict(recSys Predictor, userAndItems [][2]int) (y *mat.Dense, err error) {
	if preRanker, ok := recSys.(PreRanker); ok {
		err = preRanker.PreRank()
		if err != nil {
			log.Errorf("pre rank error: %v", err)
			return
		}
	}

	y = mat.NewDense(len(userAndItems), 1, nil)
	var x *mat.Dense
	for i, userAndItem := range userAndItems {
		userId := userAndItem[0]
		itemId := userAndItem[1]
		var (
			userFeature, itemFeature Tensor
		)
		userFeature, err = recSys.GetUserFeature(userId)
		if err != nil {
			log.Errorf("get user feature error: %v", err)
			continue
		}
		itemFeature, err = recSys.GetItemFeature(itemId)
		if err != nil {
			log.Infof("get item feature failed: %v", err)
			continue
		}
		xSlice := utils.ConcatSlice(userFeature, itemFeature)
		if i == 0 {
			x = mat.NewDense(len(userAndItems), len(xSlice), nil)
		}

		_, xCol := x.Dims()
		if len(xSlice) != xCol {
			log.Errorf("x slice length %d != x col %d", len(xSlice), xCol)
			return
		}
		x.SetRow(i, xSlice)
	}
	recSys.Predict(x, y)
	return
}

func GetSample(recSys RecSys) (sample ps.Samples, err error) {
	var (
		userFeatureWidth, itemFeatureWidth int
		userFeatureCache                   *ccache.Cache
		itemFeatureCache                   *ccache.Cache
	)
	userFeatureCache = ccache.New(
		ccache.Configure().MaxSize(userFeatureCacheSize).ItemsToPrune(userFeatureCacheSize / 100),
	)
	itemFeatureCache = ccache.New(
		ccache.Configure().MaxSize(itemFeatureCacheSize).ItemsToPrune(itemFeatureCacheSize / 100),
	)

	sampleGen, ok := recSys.(Trainer)
	if !ok {
		panic("sample generator not implemented")
	}
	sampleCh, err := sampleGen.SampleGenerator()
	if err != nil {
		panic(err)
	}

	for s := range sampleCh {
		var (
			user, item *ccache.Item
		)
		userIdStr := strconv.Itoa(s.UserId)
		user, err = userFeatureCache.Fetch(userIdStr, time.Hour*24, func() (cItem interface{}, err error) {
			cItem, err = recSys.GetUserFeature(s.UserId)
			return
		})
		if err != nil {
			continue
		}
		userFeature := user.Value().(Tensor)
		if userFeatureWidth == 0 {
			userFeatureWidth = len(userFeature)
		}
		if len(userFeature) != userFeatureWidth {
			log.Errorf("user feature length mismatch: %v:%v",
				userFeatureWidth, len(userFeature))
			continue
		}

		itemIdStr := strconv.Itoa(s.ItemId)
		item, err = itemFeatureCache.Fetch(itemIdStr, time.Hour*24, func() (cItem interface{}, err error) {
			cItem, err = recSys.GetItemFeature(s.ItemId)
			return
		})
		if err != nil {
			continue
		}
		itemFeature := item.Value().(Tensor)

		// if ItemEmbedding interface is implemented, use item embedding
		if _, ok := recSys.(ItemEmbedding); ok {
			if itemEmb, ok := itemEmbeddingMap.Get(strconv.Itoa(s.ItemId)); ok {
				itemFeature = append(itemFeature, itemEmb...)
			} else {
				var zeroItemEmb [ItemEmbDim]float64
				itemFeature = append(itemFeature, zeroItemEmb[:]...)
			}
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

func GetItemEmbeddingModelFromUb(iSeq ItemEmbedding) (mod model.Model, err error) {
	itemSeq, err := iSeq.ItemSeqGenerator()
	if err != nil {
		return
	}
	mod, err = embedding.TrainEmbedding(itemSeq, ItemEmbWindow, ItemEmbDim, 1)
	return
}
