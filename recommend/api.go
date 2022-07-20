package recommend

import (
	"github.com/gin-gonic/gin"
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
//	 curl --header "Content-Type: application/json" \
//	   --request POST \
//	   --data '{"userId":107,"itemIdList":[1,2,39]}' \
//	   http://localhost:8080/api/v1/recommend
func StartHttpApi(predict Predictor, path string, addr string) (err error) {
	engine := gin.Default()
	engine.Any(path, func(c *gin.Context) {
		//bind request to RecApiRequest
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
			scores, err := Rank(predict, req.UserId, req.ItemIdList)
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
			resp.ItemScoreList = scores
			c.JSON(200, resp)
			return
		}
	})

	return engine.Run(addr)
}
