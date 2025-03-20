from sanic import Sanic
from sanic.response import json
from sanic_ext import Extend
import os
from main import search_images

app = Sanic("server")

img_dir = os.path.abspath("images")

@app.get("imgpaths/")
async def get_image_path(request):
  files = os.listdir(img_dir)
  return json(files)

app.static("/images", img_dir)

@app.get("/search")
async def search_handler(request):
  query = request.args.get("query")
  if not query:
    return json({"error": "missing 'query'"})
  results = search_images(query)
  return json(results)

@app.middleware("response")
async def add_cors(request, response):
  response.headers["Access-Control-Allow-Origin"] = "*"
  response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
  response.headers["Access-Control-Allow-Headers"] = "Content-Type"

if __name__ == "__main__":
  app.run(host="localhost", port=8080, debug=True)