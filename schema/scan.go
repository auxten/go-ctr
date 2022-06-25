package schema

import (
	"database/sql"
)

//TableScanner could get schema or content of given mysql table
type TableScanner interface {
	GetSchema() (schema *Schema, err error)
	GetRows() (rows *sql.Rows, err error)
}
