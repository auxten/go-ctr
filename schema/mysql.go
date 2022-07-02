package schema

import (
	"database/sql"
	"errors"
	"fmt"
	"net"
	"strconv"
	"sync"

	mysql "github.com/go-sql-driver/mysql"
	"github.com/xwb1989/sqlparser"
)

var (
	ErrNotDDL      = errors.New("not a ddl statement")
	ErrDDLNotFound = errors.New("ddl not found")
)

var _ TableScanner = &MysqlScanner{}

type MysqlScanner struct {
	DbName   string
	Host     string
	Port     int
	User     string
	Password string
	conn     *sql.DB
	sync.Once
}

func ParseDsn(dsn string) (host string, port int,
	user string, password string, dbName string, err error) {
	var (
		addr    string
		portStr string
		dsnInfo *mysql.Config
	)
	if dsnInfo, err = mysql.ParseDSN(dsn); err != nil {
		return
	}
	addr = dsnInfo.Addr
	user = dsnInfo.User
	password = dsnInfo.Passwd
	if host, portStr, err = net.SplitHostPort(addr); err != nil {
		return
	}
	if portStr != "" {
		if port, err = strconv.Atoi(portStr); err != nil {
			return
		}
	} else {
		port = 3306
	}
	dbName = dsnInfo.DBName
	return
}

func NewMysqlScanner(dbName, host, user, password string, port int) *MysqlScanner {
	return &MysqlScanner{
		DbName:   dbName,
		Host:     host,
		Port:     port,
		User:     user,
		Password: password,
	}
}

func (s *MysqlScanner) initConn() (err error) {
	s.Do(func() {
		s.conn, err = sql.Open("mysql",
			s.User+":"+s.Password+"@tcp("+s.Host+":"+fmt.Sprint(s.Port)+")/"+s.DbName)
	})
	return err
}

func (s *MysqlScanner) GetSchema(tableName string) (schema *Schema, err error) {
	if err = s.initConn(); err != nil {
		return
	}

	ddl, err := s.conn.Query("SHOW CREATE TABLE " + tableName)
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
func (s *MysqlScanner) GetRows(query string) (rows *sql.Rows, err error) {
	if s.initConn() != nil {
		return
	}

	rows, err = s.conn.Query(query)
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
				Name: def.Name.String(),
				Type: def.Type.Type,
			})
			if def.Type.Comment != nil {
				schema.Columns[i].Comment = string(def.Type.Comment.Val)
			}
			if def.Type.Length != nil {
				schema.Columns[i].Size = string(def.Type.Length.Val)
			}
		}
	default:
		err = ErrNotDDL
	}

	return
}
