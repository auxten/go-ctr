package recommend

import (
	"context"
	"fmt"
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
	StageKey              = "stage"
	ItemEmbDim            = 10
	ItemEmbWindow         = 5
	UserBehaviorLen       = 10
	userFeatureCacheSize  = 200000
	itemFeatureCacheSize  = 2000000
	userBehaviorCacheSize = userFeatureCacheSize * UserBehaviorLen
)

var (
	itemEmbeddingModel model.Model
	itemEmbeddingMap   word2vec.EmbeddingMap
	DebugUserId        int
	DebugItemId        int
)

type Tensor []float64

type Stage int

const (
	TrainStage Stage = iota
	PredictStage
)

type TrainSample struct {
	Data ps.Samples
	Info SampleInfo
}

type RecSys interface {
	UserFeaturer
	ItemFeaturer
	Trainer
	FeatureOverview
}

type Predictor interface {
	UserFeaturer
	ItemFeaturer
	PredictAbstract
}

type PredictAbstract interface {
	Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense
}

type Trainer interface {
	SampleGenerator(context.Context) (<-chan Sample, error)
}

type UserFeaturer interface {
	GetUserFeature(context.Context, int) (Tensor, error)
}

//UserBehavior interface is used to get user behavior feature.
// typically, it is user's clicked/bought/liked item id list ordered by time desc.
// During training, you should limit the seq to avoid time travel,
//	maxPk or maxTs could be used here:
//	 - maxPk is the max primary key of user behavior table.
//	 - maxTs is the max timestamp of user behavior table.
//	 - maxLen is the max length of user behavior seq, if total len is
// 		greater than maxLen, the seq will be truncated from the tail.
//  	which is latest maxLen items.
// specially, -1 means no limit.
//During prediction, you should use the latest user behavior seq.
type UserBehavior interface {
	GetUserBehavior(ctx context.Context, userId int,
		maxLen int64, maxPk int64, maxTs int64) (itemSeq []int, err error)
}

type ItemFeaturer interface {
	GetItemFeature(context.Context, int) (Tensor, error)
}

// ItemEmbedding is an interface used to generate item embedding with item2vec model
// by just providing a behavior based item sequence.
// Example: user liked items sequence, user bought items sequence, user viewed items sequence
type ItemEmbedding interface {
	ItemSeqGenerator(context.Context) (<-chan string, error)
}

type SampleInfo struct {
	UserProfileRange  [2]int // [start, end)
	UserBehaviorRange [2]int // [start, end)
	ItemFeatureRange  [2]int // [start, end)
	CtxFeatureRange   [2]int // [start, end)
}

type UserItemOverview struct {
	UserId       int `json:"user_id"`
	UserFeatures map[string]interface{}
}

type ItemOverView struct {
	ItemId       int `json:"item_id"`
	ItemFeatures map[string]interface{}
}

type UserItemOverviewResult struct {
	Users []UserItemOverview `json:"users"`
}

type ItemOverviewResult struct {
	Items []ItemOverView `json:"items"`
}

type DashboardOverviewResult struct {
	Users         int `json:"users"`
	Items         int `json:"items"`
	TotalPositive int `json:"total_positive"`
	ValidPositive int `json:"valid_positive"`
	ValidNegative int `json:"valid_negative"`
}

type FeatureOverview interface {
	// GetUsersFeatureOverview returns offset and size used for paging query
	GetUsersFeatureOverview(ctx context.Context, offset, size int, opts map[string][]string) (UserItemOverviewResult, error)

	// GetItemsFeatureOverview returns offset and size used for paging query
	GetItemsFeatureOverview(ctx context.Context, offset, size int, opts map[string][]string) (ItemOverviewResult, error)

	// GetDashboardOverview returns dashboard overview, see DashboardOverviewResult
	GetDashboardOverview(ctx context.Context) (DashboardOverviewResult, error)
}

type PreRanker interface {
	PreRank(context.Context) error
}

type PreTrainer interface {
	PreTrain(context.Context) error
}

type ItemScore struct {
	ItemId int     `json:"itemId"`
	Score  float64 `json:"score"`
}

type Sample struct {
	UserId    int     `json:"userId"`
	ItemId    int     `json:"itemId"`
	Label     float64 `json:"label"`
	Timestamp int64   `json:"timestamp"`
}

func Train(ctx context.Context, recSys RecSys, mlp base.Fiter) (model Predictor, err error) {
	rand.Seed(0)
	ctx = context.WithValue(ctx, StageKey, TrainStage)

	if preTrain, ok := recSys.(PreTrainer); ok {
		err = preTrain.PreTrain(ctx)
		if err != nil {
			log.Errorf("pre train error: %v", err)
			return
		}
	}

	if itemEbd, ok := recSys.(ItemEmbedding); ok {
		itemEmbeddingModel, err = GetItemEmbeddingModelFromUb(ctx, itemEbd)
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

	trainSample, err := GetSample(recSys, ctx)
	sampleLen := len(trainSample.Data)
	sampleDense := mat.NewDense(sampleLen, len(trainSample.Data[0].Input), nil)
	for i, sample := range trainSample.Data {
		sampleDense.SetRow(i, sample.Input)
	}
	yClass := mat.NewDense(sampleLen, 1, nil)
	for i, sample := range trainSample.Data {
		yClass.Set(i, 0, sample.Response[0])
	}

	// start training
	log.Infof("\nstart training with %d samples\n", sampleLen)

	//TODO: pass trainSample to Fit??
	mlp.Fit(sampleDense, yClass)
	type modelImpl struct {
		UserFeaturer
		ItemFeaturer
		PredictAbstract
		FeatureOverview
	}
	model = &modelImpl{
		UserFeaturer:    recSys,
		ItemFeaturer:    recSys,
		PredictAbstract: mlp.(PredictAbstract),
		FeatureOverview: recSys,
	}

	return
}

func Rank(ctx context.Context, recSys Predictor, userId int, itemIds []int) (itemScores []ItemScore, err error) {
	ctx = context.WithValue(ctx, StageKey, PredictStage)
	if preRanker, ok := recSys.(PreRanker); ok {
		err = preRanker.PreRank(ctx)
		if err != nil {
			log.Errorf("pre rank error: %v", err)
			return
		}
	}
	var (
		userFeature, itemFeature Tensor
	)
	itemScores = make([]ItemScore, len(itemIds))
	userFeature, err = recSys.GetUserFeature(ctx, userId)
	if err != nil {
		log.Errorf("get user feature error: %v", err)
		return
	}
	for i, itemId := range itemIds {
		itemFeature, err = recSys.GetItemFeature(ctx, itemId)
		if err != nil {
			log.Infof("get item feature failed: %v", err)
			return
		}
		xSlice := utils.ConcatSlice(userFeature, itemFeature)
		x := mat.NewDense(1, len(xSlice), xSlice)
		y := mat.NewDense(1, 1, nil)

		score := recSys.Predict(x, y)
		itemScores[i] = ItemScore{itemId, score.At(0, 0)}
		if (DebugItemId == 0 || DebugItemId == itemIds[i]) &&
			(DebugUserId == 0 || DebugUserId == userId) {
			log.Infof("user %d: %v\nitem %d: %v\nscore %f",
				userId, userFeature, itemIds[i], itemFeature, score.At(0, 0))
		}
	}
	return
}

func BatchPredict(ctx context.Context, recSys Predictor, userAndItems [][2]int) (y *mat.Dense, err error) {
	ctx = context.WithValue(ctx, StageKey, PredictStage)
	if preRanker, ok := recSys.(PreRanker); ok {
		err = preRanker.PreRank(ctx)
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
		userFeature, err = recSys.GetUserFeature(ctx, userId)
		if err != nil {
			log.Errorf("get user feature error: %v", err)
			continue
		}
		itemFeature, err = recSys.GetItemFeature(ctx, itemId)
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

func GetSample(recSys RecSys, ctx context.Context) (sample *TrainSample, err error) {
	var (
		sampleWidth       int
		userFeatureWidth  int
		itemFeatureWidth  int
		zeroItemEmb       [ItemEmbDim]float64
		zeroUserBehaviors [ItemEmbDim * UserBehaviorLen]float64

		userFeatureCache  *ccache.Cache
		itemFeatureCache  *ccache.Cache
		userBehaviorCache *ccache.Cache
	)
	userFeatureCache = ccache.New(
		ccache.Configure().MaxSize(userFeatureCacheSize).ItemsToPrune(userFeatureCacheSize / 100),
	)
	itemFeatureCache = ccache.New(
		ccache.Configure().MaxSize(itemFeatureCacheSize).ItemsToPrune(itemFeatureCacheSize / 100),
	)
	userBehaviorCache = ccache.New(
		ccache.Configure().MaxSize(userBehaviorCacheSize).ItemsToPrune(userBehaviorCacheSize / 100),
	)

	defer func() {
		userFeatureCache.Clear()
		itemFeatureCache.Clear()
		userBehaviorCache.Clear()
	}()

	sampleGen, ok := recSys.(Trainer)
	if !ok {
		panic("sample generator not implemented")
	}
	sampleCh, err := sampleGen.SampleGenerator(ctx)
	if err != nil {
		panic(err)
	}

	sample = &TrainSample{}
	for s := range sampleCh {
		var (
			user, item *ccache.Item
		)
		userIdStr := strconv.Itoa(s.UserId)
		user, err = userFeatureCache.Fetch(userIdStr, time.Hour*24, func() (ci interface{}, err error) {
			ci, err = recSys.GetUserFeature(ctx, s.UserId)
			return
		})
		if err != nil {
			continue
		}
		userFeature := user.Value().(Tensor)
		if userFeatureWidth == 0 {
			userFeatureWidth = len(userFeature)
			sample.Info.UserProfileRange[0] = 0
			sample.Info.UserProfileRange[1] = userFeatureWidth
			sample.Info.UserBehaviorRange[0] = sample.Info.UserProfileRange[1]
			sample.Info.UserBehaviorRange[1] = sample.Info.UserProfileRange[1] + ItemEmbDim*UserBehaviorLen
			// item feature here is only embeddings
			sample.Info.ItemFeatureRange[0] = sample.Info.UserBehaviorRange[1]
			sample.Info.ItemFeatureRange[1] = sample.Info.UserBehaviorRange[1] + ItemEmbDim
		}
		if len(userFeature) != userFeatureWidth {
			log.Errorf("user feature length mismatch: %v:%v",
				userFeatureWidth, len(userFeature))
			continue
		}

		itemIdStr := strconv.Itoa(s.ItemId)
		item, err = itemFeatureCache.Fetch(itemIdStr, time.Hour*24, func() (ci interface{}, err error) {
			ci, err = recSys.GetItemFeature(ctx, s.ItemId)
			return
		})
		if err != nil {
			continue
		}
		itemFeature := item.Value().(Tensor)

		// if ItemEmbedding interface is implemented, use item embedding,
		// 	else use zero embedding.
		var (
			itemEmb       = zeroItemEmb[:]
			userBehaviors = zeroUserBehaviors[:]
		)
		if _, ok := recSys.(ItemEmbedding); ok {
			if itemEmb, ok = itemEmbeddingMap.Get(strconv.Itoa(s.ItemId)); !ok {
				itemEmb = zeroItemEmb[:]
				log.Debugf("item embedding not found: %d, using zeros", s.ItemId)
			}
			// if ItemEmbedding and UserBehavior interface are both implemented,
			// use itemSeq embeddings got from GetUserBehavior as user behavior,
			//	else use zero embedding.
			if recSysUb, ok := recSys.(UserBehavior); ok {
				ubKey := fmt.Sprintf("%d_%d", s.UserId, s.Timestamp)
				ubEmb, err := userBehaviorCache.Fetch(ubKey, time.Hour*24, func() (ci interface{}, err error) {
					itemSeq, err := recSysUb.GetUserBehavior(ctx, s.UserId, UserBehaviorLen, -1, s.Timestamp)
					if err != nil {
						return
					}
					//query items embedding, fill them into user behavior
					ubTensor := make(Tensor, ItemEmbDim*UserBehaviorLen)
					for i, itemId := range itemSeq {
						if itemEmb, ok := itemEmbeddingMap.Get(strconv.Itoa(itemId)); ok {
							copy(ubTensor[i*ItemEmbDim:], itemEmb)
						}
					}
					ci = ubTensor
					return
				})
				if err != nil {
					log.Errorf("get user behavior error: %v", err)
					continue
				}
				userBehaviors = ubEmb.Value().(Tensor)
			}
		}
		if itemFeatureWidth == 0 {
			itemFeatureWidth = len(itemFeature)
			// non embedding item feature is treated as ctx feature
			sample.Info.CtxFeatureRange[0] = sample.Info.ItemFeatureRange[1]
			sample.Info.CtxFeatureRange[1] = sample.Info.ItemFeatureRange[1] + itemFeatureWidth
		}
		if len(itemFeature) != itemFeatureWidth {
			log.Errorf("item feature length mismatch: %v:%v",
				itemFeatureWidth, len(itemFeature))
			continue
		}

		one := ps.Sample{Input: utils.ConcatSlice(userFeature, userBehaviors, itemEmb, itemFeature), Response: Tensor{s.Label}}
		if sampleWidth == 0 {
			sampleWidth = len(one.Input)
		} else {
			if len(one.Input) != sampleWidth {
				err = fmt.Errorf("sample width mismatch: %v:%v", sampleWidth, len(one.Input))
				return
			}
		}

		sample.Data = append(sample.Data, one)
		if len(sample.Data)%100 == 0 {
			log.Infof("sample size: %d", len(sample.Data))
		}
	}

	return
}

func GetItemEmbeddingModelFromUb(ctx context.Context, iSeq ItemEmbedding) (mod model.Model, err error) {
	itemSeq, err := iSeq.ItemSeqGenerator(ctx)
	if err != nil {
		return
	}
	mod, err = embedding.TrainEmbedding(itemSeq, ItemEmbWindow, ItemEmbDim, 1)
	return
}
