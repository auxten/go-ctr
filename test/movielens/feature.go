package main

import (
	"database/sql"
	"fmt"
	"os"
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
	log "github.com/sirupsen/logrus"
)

const (
	DbPath           = "../movielens.db"
	EmbModelFilePath = "model.txt"
)

var (
	db *sql.DB
)

func init() {
	var err error
	db, err = sql.Open("sqlite3", DbPath)
	if err != nil {
		panic(err)
	}
}

type RecSys interface {
	ItemSeq
	UserFeature
	ItemFeature
}

type ItemSeq interface {
	ItemSeqGenerator() <-chan string
}

type UserFeature interface {
	GetUserFeature(int) []float64
}

type ItemFeature interface {
	GetItemFeature(int) []float64
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
		defer close(ch)
		rows, err := db.Query("SELECT movieId FROM ratings r WHERE r.rating > 3.5 order by userId, timestamp")
		if err != nil {
			log.Errorf("failed to query ratings: %v", err)
			return
		}
		defer rows.Close()
		for rows.Next() {
			var movieId int
			if err = rows.Scan(&movieId); err != nil {
				log.Errorf("failed to scan movieId: %v", err)
				return
			}
			ch <- fmt.Sprintf("%d", movieId)
		}
	}()
	return ch
}

func GetItemEmbeddingModelFromUb(recSys RecSys) (mod model.Model, err error) {
	itemSeq := recSys.ItemSeqGenerator()
	mod, err = embedding.TrainEmbedding(itemSeq, 5, 10, 1)
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
		copy(top5GenresTensor[i*10:], feature.HashOneHot([]byte(genre.Key), 10))
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

func GetSample() (sample ps.Samples) {
	rows, err := db.Query("SELECT userId, movieId, rating FROM ratings")
	return
}

func (recSys *RecSysImpl) GetScore(userTensor []float64, itemTensor []float64) (score []float64) {
	return recSys.Neural.Predict(utils.ConcatSlice(userTensor, itemTensor))
}

func (recSys *RecSysImpl) PreTrain(embModelPath string) (err error) {
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
	modelFileWriter, err := os.OpenFile(embModelPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer modelFileWriter.Close()
	err = recSys.EmbeddingMod.Save(modelFileWriter, vector.Agg)
	if err != nil {
		return err
	}
	modelFileWriter.Sync()
	recSys.embModelPath = embModelPath
	return nil
}

func (recSys *RecSysImpl) Train() (err error) {
	return
}
