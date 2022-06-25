package schema

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

const (
	mysqlTestDbName    = "auxten"
	mysqlTestTableName = "task_ddl"
	mysqlTestHost      = "mysql.test"
	mysqlTestUser      = "test"
	mysqlTestPassword  = "dfb25e6d925aadea2a543b9d772665b6"
	mysqlTestPort      = 8336
)

func TestSchema(t *testing.T) {
	scaner := NewMysqlTableScanner(mysqlTestDbName, mysqlTestTableName, mysqlTestHost, mysqlTestUser, mysqlTestPassword, mysqlTestPort)
	Convey("get schema", t, func() {
		schema, err := scaner.GetSchema()
		So(err, ShouldBeNil)
		So(schema, ShouldNotBeNil)
		So(schema.DbName, ShouldEqual, "auxten")
		So(schema.TableName, ShouldEqual, "task_ddl")
		So(schema.Columns, ShouldHaveLength, 8)
	})

	Convey("schema not exist", t, func() {
		scaner := NewMysqlTableScanner(mysqlTestDbName, "not_exist", mysqlTestHost, mysqlTestUser, mysqlTestPassword, mysqlTestPort)
		schema, err := scaner.GetSchema()
		So(fmt.Sprint(err), ShouldContainSubstring, "Error 1146: Table 'auxten.not_exist' doesn't exist")
		So(schema, ShouldBeNil)
	})

	Convey("get rows", t, func() {
		rows, err := scaner.GetRows()
		So(rows.Next(), ShouldBeTrue)
		var (
			task_id     int64
			title       string
			start_date  string
			due_date    string
			status      int
			priority    int
			description string
			created_at  string
		)
		err = rows.Scan(
			&task_id, &title, &start_date, &due_date,
			&status, &priority, &description, &created_at)
		So(err, ShouldBeNil)
		So(rows, ShouldNotBeNil)
		So(task_id, ShouldEqual, 1)
		So(title, ShouldEqual, "t1")
		So(start_date, ShouldEqual, "2022-06-19")
		So(due_date, ShouldEqual, "2022-06-20")
		So(status, ShouldEqual, 0)
		So(priority, ShouldEqual, 1)
		So(description, ShouldEqual, "task no.1")
		So(created_at, ShouldEqual, "2022-06-19 23:01:47")
	})
}
