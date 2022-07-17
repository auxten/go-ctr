package featuresales

import (
	"database/sql"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/auxten/edgeRec/feature"
	"github.com/auxten/edgeRec/nn"
	"github.com/auxten/edgeRec/ps"
	"github.com/auxten/edgeRec/schema"
	"github.com/auxten/edgeRec/utils"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
)

const (
	mysqlTestDbName   = "future_sales"
	mysqlTestHost     = "mysql.test"
	mysqlTestUser     = "test"
	mysqlTestPassword = "dfb25e6d925aadea2a543b9d772665b6"
	mysqlTestPort     = 8336
	trainCount        = 10000
	testCount         = 1000
)

func featureTransform(date string, date_block_num float64, shop_id float64,
	item_id float64, item_price float64, item_category_id float64, item_name string) []float64 {
	dateParts := strings.Split(date, ".")
	day, err := strconv.Atoi(dateParts[0])
	if err != nil {
		panic(err)
	}
	month, err := strconv.Atoi(dateParts[1])
	if err != nil {
		panic(err)
	}
	year, err := strconv.Atoi(dateParts[2])
	if err != nil {
		panic(err)
	}
	time := time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC)
	weekDay := time.Weekday()
	return utils.ConcatSlice(
		feature.SimpleOneHot(day-1, 31),
		feature.SimpleOneHot(month-1, 12),
		feature.SimpleOneHot(year-2013, 3),
		feature.SimpleOneHot(int(date_block_num), 34),
		feature.SimpleOneHot(int(shop_id), 60),
		feature.SimpleOneHot(int(item_category_id), 84),
		feature.SimpleOneHot(int(weekDay), 7),
		feature.HashOneHot(utils.Float64toBytes(item_id), 10),
		[]float64{math.Log2(item_price)},
		feature.StringSplitMultiHot(item_name, " ", 100),
	)
}

// -22 ~ 2169
func outputTransform(output float64) float64 {
	return output / 20.0
}

func outputRecovery(output float64) float64 {
	return output * 20.0
}

func TestTrain(t *testing.T) {
	Convey("get schema", t, func() {
		var (
			err error
			i   int
		)

		scanner := schema.NewMysqlScanner(
			mysqlTestDbName, mysqlTestHost, mysqlTestUser, mysqlTestPassword, mysqlTestPort)
		schema, err := scanner.GetSchema("sales_train")
		So(err, ShouldBeNil)
		So(schema.Columns, ShouldHaveLength, 6)

		fmt.Printf("training data count: %d\n", trainCount)
		// training
		trainSample := make(ps.Samples, trainCount)
		{
			trainRows, err := scanner.GetRows(fmt.Sprintf(`select date, date_block_num, shop_id, s.item_id, item_price, item_category_id, item_name, item_cnt_day from sales_train s 
				left join items i on s.item_id = i.item_id limit %d`, trainCount))
			So(err, ShouldBeNil)
			for trainRows.Next() {
				var (
					date             string
					date_block_num   float64
					shop_id          float64
					item_id          float64
					item_price       float64
					item_category_id sql.NullFloat64
					item_name        sql.NullString
					item_cnt_day     float64
				)
				err = trainRows.Scan(&date, &date_block_num, &shop_id, &item_id, &item_price, &item_category_id, &item_name, &item_cnt_day)
				if err != nil {
					log.Fatal(err)
				}
				trainSample[i] = ps.Sample{
					Input:    featureTransform(date, date_block_num, shop_id, item_id, item_price, item_category_id.Float64, item_name.String),
					Response: []float64{outputTransform(item_cnt_day)}}
				i++
				// print progress
				if i%(trainCount/10) == 0 {
					fmt.Printf(".")
				}
			}
			trainRows.Close()
		}
		// 10% test data
		fmt.Printf("test data count: %d\n", testCount)
		i = 0
		testSample := make(ps.Samples, testCount)
		rows, err := scanner.GetRows(fmt.Sprintf(`select date, date_block_num, shop_id, s.item_id, item_price, item_category_id, item_name, 
			item_cnt_day from sales_train s 
				left join items i on s.item_id = i.item_id limit %d, %d`, trainCount, testCount))
		So(err, ShouldBeNil)
		for rows.Next() {
			var (
				date             string
				date_block_num   float64
				shop_id          float64
				item_id          float64
				item_price       float64
				item_category_id sql.NullFloat64
				item_name        sql.NullString
				item_cnt_day     float64
			)
			err = rows.Scan(&date, &date_block_num, &shop_id, &item_id, &item_price, &item_category_id, &item_name, &item_cnt_day)
			if err != nil {
				log.Fatal(err)
			}
			testSample[i] = ps.Sample{
				Input:    featureTransform(date, date_block_num, shop_id, item_id, item_price, item_category_id.Float64, item_name.String),
				Response: []float64{outputTransform(item_cnt_day)}}
			i++
			// print progress
			if i%(testCount/10) == 0 {
				fmt.Printf(".")
			}
		}
		rows.Close()

		rand.Seed(0)

		n := nn.NewNeural(&nn.Config{
			Inputs:     len(trainSample[0].Input),
			Layout:     []int{len(trainSample[0].Input), 64, 64, 1},
			Activation: nn.ActivationSigmoid,
			Weight:     nn.NewUniform(0.5, 0),
			Bias:       true,
		})

		//start training
		fmt.Printf("\nstart training\n")
		trainer := ps.NewTrainer(ps.NewSGD(0.01, 0.1, 0, false), 1)
		trainer.Train(n, trainSample, testSample, 2, true)

		predictRows1, err := scanner.GetRows(fmt.Sprintf(`SELECT
																date, date_block_num,
																			shop_id,
																			s.item_id,
																			item_price,
																			item_category_id,
																			item_name,
																			item_cnt_day
															FROM
															  (SELECT date, date_block_num,
																			shop_id,
																			item_id,
																			item_price,
																			item_cnt_day
															   FROM sales_train
															   where item_cnt_day >= 0
															   GROUP BY item_cnt_day) s
															LEFT JOIN items i ON s.item_id = i.item_id
															 LIMIT 100`))
		So(err, ShouldBeNil)
		for predictRows1.Next() {
			var (
				date             string
				date_block_num   float64
				shop_id          float64
				item_id          float64
				item_price       float64
				item_category_id sql.NullFloat64
				item_name        sql.NullString
				item_cnt_day     float64
			)
			err = predictRows1.Scan(&date, &date_block_num, &shop_id, &item_id, &item_price, &item_category_id, &item_name, &item_cnt_day)
			if err != nil {
				log.Fatal(err)
			}
			input := featureTransform(date, date_block_num, shop_id, item_id, item_price, item_category_id.Float64, item_name.String)
			output := n.Predict(input)
			fmt.Printf("input: %s %f %f, output: %f - %f\n", date, date_block_num, item_id, item_cnt_day, outputRecovery(output[0]))
		}

		predictRows2, err := scanner.GetRows(fmt.Sprintf(`SELECT
																date, date_block_num,
																			shop_id,
																			s.item_id,
																			item_price,
																			item_category_id,
																			item_name,
																			item_cnt_day
															FROM
																sales_train s
															LEFT JOIN items i ON s.item_id = i.item_id
															 LIMIT 100`))
		So(err, ShouldBeNil)
		for predictRows2.Next() {
			var (
				date             string
				date_block_num   float64
				shop_id          float64
				item_id          float64
				item_price       float64
				item_category_id sql.NullFloat64
				item_name        sql.NullString
				item_cnt_day     float64
			)
			err = predictRows2.Scan(&date, &date_block_num, &shop_id, &item_id, &item_price, &item_category_id, &item_name, &item_cnt_day)
			if err != nil {
				log.Fatal(err)
			}
			input := featureTransform(date, date_block_num, shop_id, item_id, item_price, item_category_id.Float64, item_name.String)
			output := n.Predict(input)
			fmt.Printf("input: %s %f %f, output: %f - %f\n", date, date_block_num, item_id, item_cnt_day, outputRecovery(output[0]))
		}
	})
}
