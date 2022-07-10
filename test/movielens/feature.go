package main

import (
	"database/sql"
	"fmt"
	"os"

	"github.com/auxten/edgeRec/feature/embedding"
	"github.com/auxten/edgeRec/feature/embedding/model"
	"github.com/auxten/edgeRec/feature/embedding/model/modelutil/vector"
	"github.com/auxten/edgeRec/utils"
	log "github.com/sirupsen/logrus"
)

const (
	DbPath           = "../movielens.db"
	EmbModelFilePath = "model.txt"
)

var (
	db           *sql.DB
	embeddingMod model.Model
)

func init() {
	var err error
	db, err = sql.Open("sqlite3", DbPath)
	if err != nil {
		panic(err)
	}
}

type RecSys interface {
	ItemSeqGenerator() <-chan string
}

type UserFeature interface {
	Tensor() (tensor []float64)
}

type ItemFeature interface {
	Tensor() (tensor []float64)
}

type ItemScore struct {
	ItemId int     `json:"itemId"`
	Score  float64 `json:"score"`
}

type RecSysImpl struct {
}

func (uf *RecSysImpl) ItemSeqGenerator() <-chan string {
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

func Recommend(userId int, itemIds []int) (itemScores []ItemScore, err error) {
	itemScores = make([]ItemScore, len(itemIds))
	userFeature := GetUserFeature(userId)
	for i, itemId := range itemIds {
		itemFeature := GetItemFeature(itemId)
		itemEmb, _ := embeddingMod.EmbeddingByWord(fmt.Sprintf("%d", itemId))
		itemTensor := utils.ConcatSlice(itemFeature.Tensor(), itemEmb)
		score := GetScore(userFeature.Tensor(), itemTensor)
		itemScores[i] = ItemScore{itemId, score}
	}
	return
}

func GetItemFeature(itemId int) (itemFeature ItemFeature) {
	return
}

func GetUserFeature(userId int) (userFeature UserFeature) {
	return
}

func GetScore(userTensor []float64, itemTensor []float64) (score float64) {
	return
}

func PreTrain(recSys RecSys) (err error) {
	embeddingMod, err = GetItemEmbeddingModelFromUb(recSys)
	if err != nil {
		return err
	}
	modelFileWriter, err := os.OpenFile(EmbModelFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer modelFileWriter.Close()
	err = embeddingMod.Save(modelFileWriter, vector.Agg)
	if err != nil {
		return err
	}
	modelFileWriter.Sync()
	return nil
}
