from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from main import search_images, images_path

app = FastAPI()

origins = [
  "http://localhost:5173"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

img_dir = images_path

@app.get("/imgpaths")
async def get_image_path():
  files = os.listdir(img_dir)
  return files

app.mount("/images", StaticFiles(directory=img_dir), name="static")

@app.get("/search")
async def search_handler(query: str, radius: int):
  if not query:
    return {"error": "missing 'query'"}
  results = search_images(query, top_k=radius)
  return results
