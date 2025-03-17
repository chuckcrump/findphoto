import lancedb
import pyarrow as pa
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Select device (use MPS for Apple Silicon)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load Hugging Face CLIP model
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name).to(device)
model = torch.compile(model)  # Optional: Optimized execution
processor = CLIPProcessor.from_pretrained(model_name)

# Initialize LanceDB
db = lancedb.connect("./clip.db")  # Creates/opens a database in a folder
table_name = "images"

# Define schema if the table does not exist
if table_name not in db.table_names():
    schema = pa.schema([
        ("vector", lancedb.vector(512)),  # CLIP embeddings
        ("path", pa.string())  # Image file path
    ])
    table = db.create_table(table_name, schema=schema)
else:
    table = db.open_table(table_name)

# Function to encode image into CLIP vector
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    return image_features.cpu().numpy().flatten()

# Function to add image to LanceDB
def add_image(image_path):
    vector = encode_image(image_path)
    table.add([{"vector": vector, "path": image_path}])
    print(f"Added {image_path} to database.")

# Example: Add images to the DB
#add_image("images/img01.JPG")
#add_image("images/img02.jpg")
#add_image("images/img03.jpg")

def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    return text_features.cpu().numpy().flatten()

def search_images(query, top_k=2):
    query_vector = encode_text(query)
    results = table.search(query_vector).limit(top_k).to_list()
    
    for res in results:
        print(f"Match: {res['path']}")

search_images("flowers")

