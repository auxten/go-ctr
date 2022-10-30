package movielens

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/auxten/edgeRec/feature"
	"github.com/auxten/edgeRec/feature/ubcache"
	rcmd "github.com/auxten/edgeRec/recommend"
	"github.com/auxten/edgeRec/utils"
	_ "github.com/mattn/go-sqlite3"
	log "github.com/sirupsen/logrus"
)

var (
	dbOnce    sync.Once
	db        *sql.DB
	yearRegex = regexp.MustCompile(`\((\d{4})\)$`)
)

func initDb(dbPath string) (err error) {
	dbOnce.Do(func() {
		db, err = sql.Open("sqlite3", fmt.Sprintf("file:%s?cache=shared&mode=ro", dbPath))
		if err != nil {
			log.Errorf("failed to open db: %v", err)
			return
		}
		db.SetMaxOpenConns(16)
	})
	return
}

type MovielensRec struct {
	DataPath   string
	SampleCnt  int
	mRatingMap map[int][2]float64
	ubcTrain   *ubcache.UserBehaviorCache
	ubcPredict *ubcache.UserBehaviorCache
}

func (recSys *MovielensRec) ItemSeqGenerator(ctx context.Context) (ret <-chan string, err error) {
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
		// predict must use the same embedding as train
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

func (recSys *MovielensRec) GetItemFeature(ctx context.Context, itemId int) (tensor rcmd.Tensor, err error) {
	// get movie avg rating and rating count
	var (
		rows *sql.Rows
	)

	rows, err = db.Query(`select "title"     itemTitle,
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
			movieYear             int
			itemTitle, itemGenres string
			avgRating, cntRating  float64
			GenreTensor           [50]float64 // 5 * 10
		)
		if err = rows.Scan(&itemTitle, &itemGenres); err != nil {
			log.Errorf("failed to scan item %d: %v", itemId, err)
			return
		}
		// regex match year from itemTitle
		yearStrSlice := yearRegex.FindStringSubmatch(itemTitle)
		if len(yearStrSlice) > 1 {
			movieYear, err = strconv.Atoi(yearStrSlice[1])
			if err != nil {
				log.Errorf("failed to parse year: %v", err)
				return
			}
		}
		// itemGenres
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
}

func (recSys *MovielensRec) GetUserFeature(ctx context.Context, userId int) (tensor rcmd.Tensor, err error) {
	var (
		tableName        string
		rows             *sql.Rows
		ugenres          sql.NullString
		avgRating        sql.NullFloat64
		cntRating        sql.NullFloat64
		top5GenresTensor [50]float64
	)
	// get stage value from ctx
	stage := ctx.Value(rcmd.StageKey).(rcmd.Stage)
	switch stage {
	case rcmd.TrainStage:
		tableName = "user_feature_train"
	case rcmd.PredictStage:
		tableName = "user_feature_test"
	default:
		panic("unknown stage")
	}

	rows, err = db.Query(fmt.Sprintf(
		`select ugenres, avgRating, cntRating from %s where userId = ?`, tableName), userId)
	if err != nil {
		log.Errorf("failed to query ratings: %v", err)
		return
	}
	defer rows.Close()
	if rows.Next() {
		if err = rows.Scan(&ugenres, &avgRating, &cntRating); err != nil {
			log.Errorf("failed to scan user: %d feature : %v", userId, err)
			return
		}
		// split genres with delimiter "|" and ","
		genreList := strings.FieldsFunc(ugenres.String, func(r rune) bool {
			return r == '|' || r == ','
		})
		//TODO: multi-hot with occurrence weight
		top5Genres := utils.TopNOccurrences(genreList, 5)
		for i, genre := range top5Genres {
			copy(top5GenresTensor[i*10:], genreFeature(genre.Key))
		}
		tensor = utils.ConcatSlice(rcmd.Tensor{avgRating.Float64 / 5., cntRating.Float64 / 100.}, top5GenresTensor[:])
		if rcmd.DebugItemId != 0 && userId == rcmd.DebugUserId {
			log.Infof("user %d: %v ", userId, tensor)
		}
		return
	} else {
		err = fmt.Errorf("userId %d not found", userId)
		log.Errorf("userId %d not found", userId)
		return
	}
}

func genreFeature(genre string) (tensor rcmd.Tensor) {
	return feature.HashOneHot([]byte(genre), 10)
}

func (recSys *MovielensRec) SampleGenerator(_ context.Context) (ret <-chan rcmd.Sample, err error) {
	sampleCh := make(chan rcmd.Sample, 10000)
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
			"SELECT userId, movieId, rating, timestamp FROM ratings_train ORDER BY timestamp, userId ASC LIMIT ?", recSys.SampleCnt)
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
				timestamp       int64
			)
			if err = rows.Scan(&userId, &movieId, &rating, &timestamp); err != nil {
				log.Errorf("failed to scan ratings: %v", err)
				return
			}
			label = BinarizeLabel(rating)
			// label = rating / 5.0

			sampleCh <- rcmd.Sample{
				UserId:    userId,
				ItemId:    movieId,
				Label:     label,
				Timestamp: timestamp,
			}
		}
	}()

	wg.Wait()
	ret = sampleCh
	return
}

func (recSys *MovielensRec) PreTrain(ctx context.Context) (err error) {
	if err = initDb(recSys.DataPath); err != nil {
		return
	}
	// get movie avg rating and rating count
	var (
		rows1 *sql.Rows
	)
	rows1, err = db.Query(`select movieId, avg(rating) avg_r, count(rating) cnt_r
                    from ratings_train
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

	// fill user behavior cache
	if recSys.ubcTrain == nil {
		recSys.ubcTrain = ubcache.NewUserBehaviorCache()
		err = PreFillUbCache(recSys.ubcTrain, "ub_train")
		if err != nil {
			log.Errorf("failed to fill ub cache: %v", err)
			return
		}
	}

	return
}

func getRows(ctx context.Context, offset, size int, table string) (*sql.Rows, error) {
	sql := fmt.Sprintf("select * from %s", table)
	if size > 0 {
		sql = fmt.Sprintf("%s limit %d offset %d", sql, size, offset)
	}
	return db.QueryContext(ctx, sql)
}

func (recSys *MovielensRec) GetUsersFeatureOverview(ctx context.Context, offset, size int, _ map[string][]string) (res rcmd.UserItemOverviewResult, err error) {
	var rows *sql.Rows
	rows, err = getRows(ctx, offset, size, "user")
	if err != nil {
		log.Errorf("query user feature fail, err: %v", err)
		return res, err
	}

	var (
		userId  int
		istrain bool
	)

	for rows.Next() {
		err = rows.Scan(&userId, &istrain)
		if err != nil {
			log.Errorf("scan error: %v", err)
			return res, err
		}
		res.Users = append(res.Users, rcmd.UserItemOverview{UserId: userId, UserFeatures: map[string]interface{}{"is_train": istrain}})
	}
	return res, nil
}

func (recSys *MovielensRec) GetItemsFeatureOverview(ctx context.Context, offset, size int, _ map[string][]string) (res rcmd.ItemOverviewResult, err error) {
	var rows *sql.Rows
	rows, err = getRows(ctx, offset, size, "movies")
	if err != nil {
		log.Errorf("query item feature fail, err: %v", err)
		return res, err
	}
	var (
		movieId int
		title   string
		genres  string
	)
	for rows.Next() {
		err = rows.Scan(&movieId, &title, &genres)
		if err != nil {
			log.Errorf("scan error: %v", err)
		}
		res.Items = append(res.Items, rcmd.ItemOverView{
			ItemId: movieId,
			ItemFeatures: map[string]interface{}{
				"title":   title,
				"generes": genres,
			},
		})
	}
	return
}

func (recSys *MovielensRec) GetDashboardOverview(ctx context.Context) (res rcmd.DashboardOverviewResult, err error) {
	for _, cur := range []struct {
		table   string
		pointer *int
	}{{
		"user",
		&res.Users,
	},
		{
			"movies",
			&res.Items,
		},
	} {
		row := db.QueryRowContext(ctx, fmt.Sprintf("select count(*) from %s", cur.table))
		if row.Err() != nil {
			log.Errorf("query %s count fail, err: %v", cur.table, err)
			return
		}
		err = row.Scan(cur.pointer)
		if err != nil {
			log.Errorf("scan %s count result fail, err: %v", cur.table, err)
			return
		}

	}
	return
}

func BinarizeLabel(rating float64) float64 {
	if rating > 3.5 {
		return 1.0
	}
	return 0.0
}
