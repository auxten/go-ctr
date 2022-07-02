package schema

import (
	"database/sql"
	"fmt"
	"sync"

	_ "github.com/mattn/go-sqlite3" //keep
)

type SqliteScanner struct {
	DbPath string
	conn   *sql.DB
	sync.Once
}

func NewSqliteScanner(dbPath string) *SqliteScanner {
	return &SqliteScanner{
		DbPath: dbPath,
	}
}

func (s *SqliteScanner) initConn() (err error) {
	s.Do(func() {
		s.conn, err = sql.Open("sqlite3",
			fmt.Sprintf("file:%s?cache=shared", s.DbPath),
		)
	})
	return err
}

func (s *SqliteScanner) GetSchema(tableName string) (schema *Schema, err error) {
	if s.initConn() != nil {
		return
	}

	ddl, err := s.conn.Query(fmt.Sprintf("PRAGMA table_info(%s)", tableName))
	if err != nil {
		return
	}
	defer ddl.Close()
	var (
		cid                                sql.NullInt64
		name, typeStr, notNull, defaultVal sql.NullString
		pk                                 sql.NullInt64
	)
	schema = &Schema{
		TableName: tableName,
		Columns:   make([]Column, 0),
	}
	for ddl.Next() {
		if err = ddl.Scan(&cid, &name, &typeStr, &notNull, &defaultVal, &pk); err != nil {
			return
		}
		schema.Columns = append(schema.Columns, Column{
			Name: name.String,
			Type: typeStr.String,
			Size: "",
			Extra: fmt.Sprintf("notNull:%s, defaultVal:%s, pk:%d",
				notNull.String, defaultVal.String, pk.Int64),
			Comment: "",
		})
	}

	return
}

func (s *SqliteScanner) GetRows(query string) (rows *sql.Rows, err error) {
	if s.initConn() != nil {
		return
	}

	rows, err = s.conn.Query(query)
	return
}
