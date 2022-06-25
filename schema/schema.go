package schema

// Schema contains database table info
type Schema struct {
	DbName    string
	TableName string
	Columns   []Column
}

// Column contains column type, name and comment info
type Column struct {
	Name    string
	Type    string
	Size    string
	Comment string
	Extra   string
}
