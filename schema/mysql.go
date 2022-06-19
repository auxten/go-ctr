package schema

import (
	"database/sql"
	"errors"
	"fmt"
	"sync"

	_ "github.com/go-sql-driver/mysql"
	"github.com/xwb1989/sqlparser"
)

var (
	ErrNotDDL      = errors.New("not a ddl statement")
	ErrDDLNotFound = errors.New("ddl not found")
)

type MysqlTableScanner struct {
	DbName    string
	TableName string
	Host      string
	Port      int
	User      string
	Password  string
	conn      *sql.DB
	sync.Once
}

func NewMysqlTableScanner(dbName, tableName, host, user, password string, port int) *MysqlTableScanner {
	return &MysqlTableScanner{
		DbName:    dbName,
		TableName: tableName,
		Host:      host,
		Port:      port,
		User:      user,
		Password:  password,
	}
}

func (s *MysqlTableScanner) initConn() (err error) {
	s.Do(func() {
		s.conn, err = sql.Open("mysql",
			s.User+":"+s.Password+"@tcp("+s.Host+":"+fmt.Sprint(s.Port)+")/"+s.DbName)
	})
	return err
}

func (s *MysqlTableScanner) GetSchema() (schema *Schema, err error) {
	if err = s.initConn(); err != nil {
		return
	}

	ddl, err := s.conn.Query("SHOW CREATE TABLE " + s.TableName)
	if err != nil {
		return
	}
	defer ddl.Close()
	var (
		ddlStr      string
		ddlTableStr string
	)
	if ddl.Next() {
		if err = ddl.Scan(&ddlTableStr, &ddlStr); err != nil {
			return
		}

		//parse mysql ddl string
		schema, err = ParseMysqlDDL(s.DbName, ddlStr)
	} else {
		err = ddl.Err()
		if err != nil {
			return
		} else {
			err = ErrDDLNotFound
		}
	}

	return
}

// GetRows returns a row Scanner for the given table.
func (s *MysqlTableScanner) GetRows() (rows *sql.Rows, err error) {
	if s.initConn() != nil {
		return
	}

	rows, err = s.conn.Query("SELECT * FROM " + s.TableName)
	return
}

func ParseMysqlDDL(dbName string, ddlStr string) (schema *Schema, err error) {
	schema = &Schema{}
	//parse mysql ddl string
	stmt, err := sqlparser.Parse(ddlStr)
	if err != nil {
		return
	}
	switch stmt := stmt.(type) {
	case *sqlparser.DDL:
		if stmt.Action != "create" {
			err = ErrNotDDL
			return
		}
		schema.DbName = dbName
		schema.TableName = stmt.NewName.Name.String()
		for i, def := range stmt.TableSpec.Columns {
			schema.Columns = append(schema.Columns, Column{
				Name:    def.Name.String(),
				Type:    def.Type.Type,
				Comment: string(def.Type.Comment.Val),
			})
			if def.Type.Length != nil {
				schema.Columns[i].Size = string(def.Type.Length.Val)
			}
		}
	default:
		err = ErrNotDDL
	}

	return
}
