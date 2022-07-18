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
	"time"

	"github.com/auxten/edgeRec/feature"
	"github.com/auxten/edgeRec/feature/embedding"
	"github.com/auxten/edgeRec/feature/embedding/model"
	"github.com/auxten/edgeRec/feature/embedding/model/modelutil/vector"
	"github.com/auxten/edgeRec/feature/embedding/model/word2vec"
	"github.com/auxten/edgeRec/nn/base"
	rcmd "github.com/auxten/edgeRec/recommend"
	"github.com/auxten/edgeRec/utils"
	"github.com/karlseguin/ccache/v2"
	_ "github.com/mattn/go-sqlite3"
	log "github.com/sirupsen/logrus"
)

const (
	DbPath           = "../movielens.db"
	EmbModelFilePath = "model.txt"
	ItemEmbDim       = 10
	ItemEmbWindow    = 5
	SampleCnt        = 10000
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

type RecSysImpl struct {
	EmbeddingMod     model.Model
	EmbeddingMap     word2vec.EmbeddingMap
	embeddingMapOnce sync.Once
	embModelPath     string
	Neural           base.Predicter
	userFeatureCache *ccache.Cache
	itemFeatureCache *ccache.Cache
}

func (recSys *RecSysImpl) ItemSeqGenerator() (ret <-chan string, err error) {
	var (
		wg sync.WaitGroup
	)
	wg.Add(1)
	ch := make(chan string, 100)
	go func() {
		var (
			i    int
			rows *sql.Rows
		)
		defer func() {
			log.Debugf("item seq generator finished: %d", i)
			close(ch)
		}()
		rows, err = db.Query("SELECT movieId FROM ratings r WHERE r.rating > 3.5 order by userId, timestamp")
		if err != nil {
			log.Errorf("failed to query ratings: %v", err)
			wg.Done()
			return
		}
		wg.Done()
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

	wg.Wait()
	ret = ch
	return
}

func GetItemEmbeddingModelFromUb(iSeq rcmd.ItemSequencer) (mod model.Model, err error) {
	itemSeq, err := iSeq.ItemSeqGenerator()
	if err != nil {
		return
	}
	mod, err = embedding.TrainEmbedding(itemSeq, ItemEmbWindow, ItemEmbDim, 1)
	return
}

func (recSys *RecSysImpl) GetItemFeature(itemId int) (tensor rcmd.Tensor) {
	itemIdStr := strconv.Itoa(itemId)
	item, err := recSys.itemFeatureCache.Fetch(itemIdStr, time.Hour*24, func() (cItem interface{}, err error) {
		rows, err := db.Query(`select "movieId"   itemId,
						   "title"       itemTitle,
						   "genres"      itemGenres
					from movies WHERE movieId = ?`, itemId)
		if err != nil {
			log.Errorf("failed to query ratings: %v", err)
			return
		}
		defer rows.Close()
		if rows.Next() {
			var (
				itemId, movieYear     int
				itemTitle, itemGenres string
				GenreTensor           [50]float64 // 5 * 10
				itemEmb               rcmd.Tensor
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

			cItem = rcmd.Tensor(utils.ConcatSlice(tensor, GenreTensor[:], rcmd.Tensor{
				float64(movieYear-1990) / 20.0,
			}))
			return
		} else {
			return nil, fmt.Errorf("itemId %d not found", itemId)
		}
	})
	if err != nil {
		log.Errorf("failed to get item feature: %v", err)
		return
	}

	return item.Value().(rcmd.Tensor)
}

func (recSys *RecSysImpl) GetUserFeature(userId int) (tensor rcmd.Tensor) {
	userIdStr := strconv.Itoa(userId)
	user, err := recSys.userFeatureCache.Fetch(userIdStr, time.Hour*24, func() (cItem interface{}, err error) {
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

		cItem = rcmd.Tensor(utils.ConcatSlice(rcmd.Tensor{avgRating / 5., float64(cntRating) / 100.}, top5GenresTensor[:]))
		return
	})
	if err != nil {
		log.Errorf("failed to fetch user feature: %v", err)
		return
	}

	return user.Value().(rcmd.Tensor)
}

func genreFeature(genre string) (tensor rcmd.Tensor) {
	return feature.HashOneHot([]byte(genre), 10)
}

func (recSys *RecSysImpl) SampleGenerator() (ret <-chan rcmd.Sample, err error) {
	sampleCh := make(chan rcmd.Sample)
	var (
		wg sync.WaitGroup
	)
	wg.Add(1)
	go func() {
		var (
			i    int
			rows *sql.Rows
		)
		defer func() {
			log.Debugf("sample generator finished: %d", i)
			close(sampleCh)
		}()

		rows, err = db.Query(
			"SELECT userId, movieId, rating FROM ratings ORDER BY timestamp, userId ASC LIMIT ?", SampleCnt)
		if err != nil {
			log.Errorf("failed to query ratings: %v", err)
			wg.Done()
			return
		}
		wg.Done()
		defer rows.Close()
		for rows.Next() {
			i++
			var (
				userId, movieId int
				rating, label   float64
			)
			if err = rows.Scan(&userId, &movieId, &rating); err != nil {
				log.Errorf("failed to scan ratings: %v", err)
				return
			}
			if rating > 3.5 {
				label = 1.0
			} else {
				label = 0.0
			}

			sampleCh <- rcmd.Sample{
				UserId: userId,
				ItemId: movieId,
				Label:  label,
			}
		}
	}()

	wg.Wait()
	ret = sampleCh
	return
}

func (recSys *RecSysImpl) PreTrain() (err error) {
	rand.Seed(0)
	recSys.userFeatureCache = ccache.New(ccache.Configure().MaxSize(100000).ItemsToPrune(1000))
	recSys.itemFeatureCache = ccache.New(ccache.Configure().MaxSize(1000000).ItemsToPrune(10000))

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

func (recSys *RecSysImpl) PreRank() (err error) {
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
	return err
}
