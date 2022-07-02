package config

type Config struct {
	DbType string `json:"db_type"` // mysql, sqlite
	Dsn    string `json:"dsn"`
}
