package main

import (
	"database/sql"
	"fmt"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/auxten/edgeRec/feature"
	"github.com/auxten/edgeRec/feature/embedding"
	"github.com/auxten/edgeRec/feature/embedding/model"
	"github.com/auxten/edgeRec/feature/embedding/model/modelutil/vector"
	"github.com/auxten/edgeRec/feature/embedding/model/word2vec"
	"github.com/auxten/edgeRec/nn"
	"github.com/auxten/edgeRec/ps"
	"github.com/auxten/edgeRec/utils"
	_ "github.com/mattn/go-sqlite3"
	log "github.com/sirupsen/logrus"
)

const (
	DbPath           = "../movielens.db"
	EmbModelFilePath = "model.txt"
	ItemEmbDim       = 10
)

var (
	db        *sql.DB
	yearRegex = regexp.MustCompile(`\((\d{4})\)$`)
)

func init() {
	var err error
	db, err = sql.Open("sqlite3", DbPath)
	if err != nil {
		panic(err)
	}
}

type RecSys interface {
	UserFeaturer
	ItemFeaturer
}

type UserFeaturer interface {
	GetUserFeature(int) []float64
}

type ItemFeaturer interface {
	GetItemFeature(int) []float64
}

type PreTrainer interface {
	ItemSequencer
	PreTrain() error
}

type ItemSequencer interface {
	ItemSeqGenerator() <-chan string
}

type ItemScore struct {
	ItemId int     `json:"itemId"`
	Score  float64 `json:"score"`
}

type RecSysImpl struct {
	EmbeddingMod     model.Model
	EmbeddingMap     word2vec.EmbeddingMap
	embeddingMapOnce sync.Once
	embModelPath     string
	Neural           *nn.Neural
}

func (recSys *RecSysImpl) ItemSeqGenerator() <-chan string {
	ch := make(chan string, 100)
	go func() {
		var i int
		defer func() {
			log.Debugf("item seq generator finished: %d", i)
			close(ch)
		}()
		rows, err := db.Query("SELECT movieId FROM ratings r WHERE r.rating > 3.5 order by userId, timestamp")
		if err != nil {
			log.Errorf("failed to query ratings: %v", err)
			return
		}
		defer rows.Close()
		for rows.Next() {
			i++
			var movieId sql.NullInt64
			if err = rows.Scan(&movieId); err != nil {
				log.Errorf("failed to scan movieId: %v", err)
				continue
			}
			ch <- fmt.Sprintf("%d", movieId.Int64)
		}
	}()
	return ch
}

func GetItemEmbeddingModelFromUb(recSys ItemSequencer) (mod model.Model, err error) {
	itemSeq := recSys.ItemSeqGenerator()
	mod, err = embedding.TrainEmbedding(itemSeq, 5, ItemEmbDim, 1)
	return
}

func (recSys *RecSysImpl) Rank(userId int, itemIds []int) (itemScores []ItemScore, err error) {
	recSys.embeddingMapOnce.Do(func() {
		if recSys.EmbeddingMod == nil {
			embReader, err := os.Open(recSys.embModelPath)
			if err != nil {
				log.Errorf("failed to open embedding model: %v", err)
				return
			}
			defer embReader.Close()
			recSys.EmbeddingMap, err = word2vec.LoadEmbeddingMap(embReader)
			if err != nil {
				log.Errorf("failed to load embedding model: %v", err)
				return
			}
		} else {
			recSys.EmbeddingMap, err = recSys.EmbeddingMod.GenEmbeddingMap()
			if err != nil {
				return
			}
		}
	})
	itemScores = make([]ItemScore, len(itemIds))
	userFeature := recSys.GetUserFeature(userId)
	for i, itemId := range itemIds {
		itemFeature := recSys.GetItemFeature(itemId)
		score := recSys.GetScore(userFeature, itemFeature)
		itemScores[i] = ItemScore{itemId, score[0]}
	}
	return
}

func (recSys *RecSysImpl) GetItemFeature(itemId int) (tensor []float64) {
	rows, err := db.Query(`select "movieId"   itemId,
						   "title"       itemTitle,
						   "genres"      itemGenres
					from movies WHERE movieId = ?`, itemId)
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows.Close()
	for rows.Next() {
		var (
			itemId, movieYear     int
			itemTitle, itemGenres string
			GenreTensor           [50]float64 // 5 * 10
			itemEmb               []float64
			ok                    bool
		)
		if err = rows.Scan(&itemId, &itemTitle, &itemGenres); err != nil {
			log.Errorf("failed to scan movieId: %v", err)
			return
		}
		if itemEmb, ok = recSys.EmbeddingMap.Get(fmt.Sprint(itemId)); ok {
			tensor = append(tensor, itemEmb...)
		} else {
			var zeroItemEmb [ItemEmbDim]float64
			tensor = append(tensor, zeroItemEmb[:]...)
		}
		//regex match year from itemTitle
		yearStrSlice := yearRegex.FindStringSubmatch(itemTitle)
		if len(yearStrSlice) > 1 {
			movieYear, err = strconv.Atoi(yearStrSlice[1])
			if err != nil {
				log.Errorf("failed to parse year: %v", err)
				return
			}
		}
		//itemGenres
		genres := strings.Split(itemGenres, "|")
		for i, genre := range genres {
			if i >= 5 {
				break
			}
			copy(GenreTensor[i*10:], genreFeature(genre))
		}

		tensor = utils.ConcatSlice(tensor, GenreTensor[:], []float64{
			float64(movieYear-1990) / 20.0,
		})

	}
	return
}

func (recSys *RecSysImpl) GetUserFeature(userId int) (tensor []float64) {
	rows, err := db.Query(`select 
                           group_concat(genres) as ugenres
                    from ratings r2
                             left join movies t2 on r2.movieId = t2.movieId
                    where userId = ? and
                    		r2.rating > 3.5
                    group by userId`, userId)
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows.Close()
	var (
		genres           string
		avgRating        float64
		cntRating        int
		top5GenresTensor [50]float64
	)
	for rows.Next() {
		if err = rows.Scan(&genres); err != nil {
			log.Errorf("failed to scan movieId: %v", err)
			return
		}
	}

	genreList := strings.Split(genres, ",|")
	top5Genres := utils.TopNOccurrences(genreList, 5)
	for i, genre := range top5Genres {
		copy(top5GenresTensor[i*10:], genreFeature(genre.Key))
	}

	rows2, err := db.Query(`select avg(rating) as avgRating, 
						   count(rating) cntRating
					from ratings where userId = ?`, userId)
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows2.Close()
	for rows2.Next() {
		if err = rows2.Scan(&avgRating, &cntRating); err != nil {
			log.Errorf("failed to scan movieId: %v", err)
			return
		}
	}

	tensor = utils.ConcatSlice([]float64{avgRating, float64(cntRating)}, top5GenresTensor[:])

	return
}

func genreFeature(genre string) (tensor []float64) {
	return feature.HashOneHot([]byte(genre), 10)
}

func GetSample(recSys RecSys) (sample ps.Samples) {
	rows, err := db.Query("SELECT userId, movieId, rating FROM ratings ORDER BY timestamp ASC LIMIT 2000")
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows.Close()
	var (
		userFeatureWidth, itemFeatureWidth int
	)
	for rows.Next() {
		var (
			userId, movieId int
			rating          float64
			label           float64
		)
		if err = rows.Scan(&userId, &movieId, &rating); err != nil {
			log.Errorf("failed to scan movieId: %v", err)
			return
		}
		userFeature := recSys.GetUserFeature(userId)
		itemFeature := recSys.GetItemFeature(movieId)
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

		if rating > 3.5 {
			label = 1.0
		} else {
			label = 0.0
		}
		sample = append(sample, ps.Sample{Input: utils.ConcatSlice(userFeature, itemFeature), Response: []float64{label}})
		if len(sample)%100 == 0 {
			log.Infof("sample size: %d", len(sample))
		}
	}
	return
}

func (recSys *RecSysImpl) GetScore(userTensor []float64, itemTensor []float64) (score []float64) {
	return recSys.Neural.Predict(utils.ConcatSlice(userTensor, itemTensor))
}

func (recSys *RecSysImpl) PreTrain() (err error) {
	recSys.EmbeddingMod, err = GetItemEmbeddingModelFromUb(recSys)
	if err != nil {
		return err
	}
	recSys.embeddingMapOnce.Do(func() {
		recSys.EmbeddingMap, err = recSys.EmbeddingMod.GenEmbeddingMap()
		if err != nil {
			return
		}
	})
	modelFileWriter, err := os.OpenFile(EmbModelFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer modelFileWriter.Close()
	err = recSys.EmbeddingMod.Save(modelFileWriter, vector.Agg)
	if err != nil {
		return err
	}
	modelFileWriter.Sync()
	recSys.embModelPath = EmbModelFilePath
	return nil
}

func Train(recSys RecSys) (err error) {
	rand.Seed(0)
	if preTrain, ok := recSys.(PreTrainer); ok {
		err = preTrain.PreTrain()
		if err != nil {
			return
		}
	}
	trainSample := GetSample(recSys)
	sampleLen := len(trainSample)
	neural := nn.NewNeural(&nn.Config{
		Inputs:     len(trainSample[0].Input),
		Layout:     []int{len(trainSample[0].Input), 64, 64, 1},
		Activation: nn.ActivationSigmoid,
		Weight:     nn.NewUniform(0.5, 0),
		Bias:       true,
	})

	//start training
	fmt.Printf("\nstart training with %d samples\n", sampleLen)
	trainer := ps.NewTrainer(ps.NewSGD(0.01, 0.1, 0, false), 1)
	trainer.Train(neural, trainSample[:8*sampleLen/10], trainSample[8*sampleLen/10:], 10, true)

	recSys.(*RecSysImpl).Neural = neural
	return
}
