package schema

import (
	"database/sql"
	"os"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSqliteScanner(t *testing.T) {
	//create temp db
	tempDbFile, err := os.CreateTemp("", "sqlite_test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tempDbFile.Name())
	db, err := sql.Open("sqlite3", tempDbFile.Name())
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec("CREATE TABLE task_ddl (" +
		"task_id INTEGER PRIMARY KEY, title TEXT, start_date TEXT, due_date TEXT, " +
		"status INT, priority INT, description TEXT, created_at TEXT)")
	if err != nil {
		t.Fatal(err)
	}

	// insert some data
	_, err = db.Exec("INSERT INTO task_ddl (task_id, title, start_date, due_date, status, priority, description, created_at) " +
		"VALUES (1, 't1', '2022-06-19', '2022-06-20', 0, 1, 'd1', '2022-06-19'), " +
		"(2, 't2', '2022-06-19', '2022-06-20', 0, 1, 'd2', '2022-06-19')")
	if err != nil {
		t.Fatal(err)
	}

	_ = db.Close()
	Convey("get schema", t, func() {
		scanner := NewSqliteScanner(tempDbFile.Name())
		schema, err := scanner.GetSchema("task_ddl")
		So(err, ShouldBeNil)
		So(schema.Columns, ShouldHaveLength, 8)
	})

	Convey("get rows", t, func() {
		scanner := NewSqliteScanner(tempDbFile.Name())
		rows, err := scanner.GetRows("select * from task_ddl")
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
		So(description, ShouldEqual, "d1")
		So(created_at, ShouldEqual, "2022-06-19")
	})
}
