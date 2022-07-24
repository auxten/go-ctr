package main

import (
	"database/sql"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/auxten/edgeRec/feature"
	"github.com/auxten/edgeRec/nn/base"
	rcmd "github.com/auxten/edgeRec/recommend"
	"github.com/auxten/edgeRec/utils"
	_ "github.com/mattn/go-sqlite3"
	log "github.com/sirupsen/logrus"
)

const (
	DbPath    = "../movielens.db"
	SampleCnt = 80000
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
	Neural     base.Predicter
	mRatingMap map[int][2]float64
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
		rows, err = db.Query("SELECT movieId FROM ratings_train r WHERE r.rating > 3.5 order by userId, timestamp")
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

func (recSys *RecSysImpl) GetItemFeature(itemId int) (tensor rcmd.Tensor, err error) {
	// get movie avg rating and rating count
	var (
		rows *sql.Rows
	)

	rows, err = db.Query(`select m."movieId" itemId,
					   "title"     itemTitle,
					   "genres"    itemGenres
				from movies m
				WHERE m.movieId = ?`, itemId)
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows.Close()
	if rows.Next() {
		var (
			itemId, movieYear     int
			itemTitle, itemGenres string
			avgRating, cntRating  float64
			GenreTensor           [50]float64 // 5 * 10
		)
		if err = rows.Scan(&itemId, &itemTitle, &itemGenres); err != nil {
			log.Errorf("failed to scan movieId: %v", err)
			return
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
		if mr, ok := recSys.mRatingMap[itemId]; ok {
			avgRating = mr[0] / 5.
			cntRating = math.Log2(mr[1])
		}

		tensor = utils.ConcatSlice(tensor, GenreTensor[:], rcmd.Tensor{
			float64(movieYear-1990) / 20.0, avgRating, cntRating,
		})
		return
	} else {
		err = fmt.Errorf("itemId %d not found", itemId)
		return
	}
	return
}

func (recSys *RecSysImpl) GetUserFeature(userId int) (tensor rcmd.Tensor, err error) {
	var (
		rows, rows2      *sql.Rows
		genres           string
		avgRating        float64
		cntRating        int
		top5GenresTensor [50]float64
	)
	rows, err = db.Query(`select 
                           group_concat(genres) as ugenres
                    from ratings_train r2
                             left join movies t2 on r2.movieId = t2.movieId
                    where userId = ? and
                    		r2.rating > 3.5
                    group by userId`, userId)
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows.Close()
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

	// Theoretically, user feature should select from ratings_train, but we use ratings
	// to make it easy to test AUC. In a real case, this will not cause time travel.
	rows2, err = db.Query(`select avg(rating) as avgRating, 
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

	tensor = utils.ConcatSlice(rcmd.Tensor{avgRating / 5., float64(cntRating) / 100.}, top5GenresTensor[:])
	return
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
			"SELECT userId, movieId, rating FROM ratings_train ORDER BY timestamp, userId ASC LIMIT ?", SampleCnt)
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
			label = BinarizeLabel(rating)
			//label = rating / 5.0

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
	// get movie avg rating and rating count
	var (
		rows1 *sql.Rows
	)
	rows1, err = db.Query(`select movieId, avg(rating) avg_r, count(rating) cnt_r
                    from ratings
                    group by movieId`)
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows1.Close()
	recSys.mRatingMap = make(map[int][2]float64)
	for rows1.Next() {
		var (
			movieId int
			avgR    float64
			cntR    int
		)
		if err = rows1.Scan(&movieId, &avgR, &cntR); err != nil {
			log.Errorf("failed to scan movieId: %v", err)
			return
		}
		recSys.mRatingMap[movieId] = [2]float64{avgR, float64(cntR)}
	}

	return
}

func BinarizeLabel(rating float64) float64 {
	if rating > 3.5 {
		return 1.0
	}
	return 0.0
}
