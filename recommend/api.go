package recommend

import (
	"embed"
	"github.com/gin-gonic/gin"
	"io/fs"
	"net/http"
	"strconv"
)

type RecApiRequest struct {
	UserId     int   `json:"userId"`
	ItemIdList []int `json:"itemIdList"`
}

type RecApiResponse struct {
	ItemScoreList []ItemScore `json:"itemScoreList"`
}

// StartHttpApi starts the http api for recommendation
// Query by:
//
//	curl --header "Content-Type: application/json" \
//	  --request POST \
//	  --data '{"userId":107,"itemIdList":[1,2,39]}' \
//	  http://localhost:8080/api/v1/recommend
func StartHttpApi(predict Predictor, path string, addr string, efs *embed.FS) (err error) {
	engine := gin.Default()
	engine.GET("/service/useritems", func(c *gin.Context) {
		querys := c.Request.URL.Query()

		var size = 0
		if data := querys.Get("size"); data != "" {
			i, err := strconv.Atoi(data)
			if err == nil {
				size = i
			}
		}

		var offset = 0
		if data := querys.Get("page"); data != "" {
			i, err := strconv.Atoi(data)
			if err == nil && size > 0 && i > 0 {
				offset = (i - 1) * size
			}
		}

		if overview, ok := predict.(FeatureOverview); ok {
			users, err := overview.GetUsersFeatureOverview(c, offset, size, querys)
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
			c.JSON(200, users)
		} else {
			c.JSON(200, "do not support feature overview")
		}
		return
	})

	engine.GET("/service/items", func(c *gin.Context) {
		querys := c.Request.URL.Query()
		var size = 0
		if data := querys.Get("size"); data != "" {
			i, err := strconv.Atoi(data)
			if err == nil {
				size = i
			}
		}

		var offset = 0
		if data := querys.Get("page"); data != "" {
			i, err := strconv.Atoi(data)
			if err == nil && size > 0 && i > 0 {
				offset = (i - 1) * size
			}
		}

		if overview, ok := predict.(FeatureOverview); ok {
			users, err := overview.GetItemsFeatureOverview(c, offset, size, querys)
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
			c.JSON(200, users)
		} else {
			c.JSON(200, "do not support item overview")
		}
		return
	})

	engine.GET("/service/overview", func(c *gin.Context) {
		if overview, ok := predict.(FeatureOverview); ok {
			users, err := overview.GetDashboardOverview(c)
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
			c.JSON(200, users)
		} else {
			c.JSON(200, "do not support overview")
		}
		return
	})

	engine.Any(path, func(c *gin.Context) {
		// bind request to RecApiRequest
		var (
			req RecApiRequest
		)
		if err := c.ShouldBind(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		if len(req.ItemIdList) == 0 {
			// todo: some default recall algorithm
			c.JSON(400, gin.H{"error": "itemIdList is empty"})
			return
		} else {
			resp := RecApiResponse{}
			// get features in request from gin Context
			scores, err := Rank(c, predict, req.UserId, req.ItemIdList)
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
			resp.ItemScoreList = scores
			c.JSON(200, resp)
			return
		}
	})
	var assetsFs, rootFs fs.FS
	assetsFs, err = fs.Sub(efs, "frontend/website/assets")
	if err != nil {
		panic(err)
	}
	rootFs, err = fs.Sub(efs, "frontend/website")
	if err != nil {
		panic(err)
	}

	engine.StaticFileFS("/favicon.ico", "favicon.ico", http.FS(rootFs))
	engine.StaticFS("/assets", http.FS(assetsFs))
	engine.Any("/", func(c *gin.Context) {
		c.FileFromFS("", http.FS(rootFs))
	})
	engine.GET("index.html", func(c *gin.Context) {
		file, _ := efs.ReadFile("frontend/website/index.html")
		c.Data(http.StatusOK, "text/html", file)
	})

	return engine.Run(addr)
}
