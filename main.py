import lancedb
import pyarrow as pa
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from os import listdir
from os.path import isfile, join

# REQUIRED
images_path = "/home/andy/Pictures/Thomas_Cole"

compute_device = "cpu"
if torch.cuda.is_available():
   compute_device = "cuda"
elif torch.backends.mps.is_available():
   compute_device = "mps"

device = compute_device 
print(device)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
#model = torch.compile(model) #Optional
processor = CLIPProcessor.from_pretrained(model_name)

db = lancedb.connect("./clip.db")
table_name = "images"

schema = pa.schema([
    ("vector", lancedb.vector(512)),
    ("path", pa.string())
])

tables = db.table_names()

if table_name in tables:
  table = db.open_table(table_name)
else:
  table = db.create_table(table_name, schema=schema, mode="overwrite")

# index if your db is large enough
# table.create_index(num_partitions=8, num_sub_vectors=8)

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if width < 1000 or height < 1000:
        return None
    width, height = image.size
    image = image.resize((int(width // 1.33333), int(height // 1.33333)))

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    vector = image_features.cpu().numpy().flatten()
    normal = np.linalg.norm(vector)
    return vector / normal if normal > 0 else vector

def add_images(path):
    photos = [file for file in listdir(path) if isfile(join(path, file))]
    existing_photos = set(table.to_arrow()["path"].to_pylist())
    for photo in photos:
      img_path = join(path, photo)
      if img_path in existing_photos:
          continue
      vector = encode_image(img_path)
      table.add([{"vector": vector, "path": img_path}])
      print(f"Added {img_path} to database.")

add_images(images_path)

def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    vector = text_features.cpu().numpy().flatten()
    normal = np.linalg.norm(vector)
    return vector / normal if normal > 0 else vector

def search_images(query, top_k=3):
    query_vector = encode_text(query)
    results = table.search(query_vector).limit(top_k).to_list()
    return [{"path": res["path"], "file_name": res["path"].split("/")[-1]} for res in results]