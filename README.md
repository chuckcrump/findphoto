# FindPhoto

### Find your crap faster

FindPhoto is an image context search app using openai's CLIP model

### Setup

- Create venv `python3 -m venv .venv`
- Active venv `source .venv/bin/activate` or other for your current shell
- Install packages `pip install -r requirements.txt`
- Finally install `PyTorch` for your platform get it [here](https://pytorch.org/get-started/locally/)
- You can change the model in the `model_name`
- Then set the `images_path` in `main.py` to the path to be processed
- Finally run `python3 server.py` and it's running at http://localhost:8000
### Future

- Electron app
