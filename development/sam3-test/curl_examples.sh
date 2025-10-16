export API_KEY=EQ8CBtIdyAMC4KZuflxx

#embed image
curl -X POST http://localhost:9001/sam3/embed_image?api_key=$API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "image": {
      "type": "url",
      "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg"
    }
  }'

# returns e.g. {"image_id":"9e81f8c2e3e9","time":0.6901539321988821}
# image id can be passed to visual segment request to use cached embedding for super fast point/box segmentation



#visual segment
#optionally pass image_id to use cached embedding for super fast point/box segmentation
curl -X POST http://localhost:9001/sam3/visual_segment?api_key=$API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "image": {
      "type": "url",
      "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg"
    },
    "prompts": [{"points": [{"x": 100, "y": 100, "positive": true}]}],
    "format": "rle"
  }'





