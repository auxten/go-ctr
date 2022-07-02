package schema

import (
	"database/sql"
)

//TableScanner could get schema or content of given mysql table
type TableScanner interface {
	GetSchema(tableName string) (schema *Schema, err error)
	GetRows(query string) (rows *sql.Rows, err error)
}
