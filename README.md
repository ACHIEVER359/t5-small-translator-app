# AI Translator

A small Gradio web app based on the `google-t5/t5-small` model from Hugging Face.

## Run locally

Requires Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows Command Prompt**
```cmd
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

The first launch downloads the model, so it can take a while and needs an internet connection.

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Gradio** as the SDK.
3. Upload or push `app.py`, `requirements.txt`, and `README.md` into the Space repository.
4. The Space will install dependencies and build the app automatically.

## Push this project to GitHub

Create an empty repository on https://github.com/new, then run these commands in this folder (replace `YOUR-USERNAME` and `YOUR-REPO`):

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Create AI translator app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

If this folder is already a Git repository, skip `git init`. If `origin` already exists, check it with `git remote -v` before adding or changing it.

## Important model limitation

`google-t5/t5-small` is not fine-tuned specifically for translation. The app preserves the model and language prompt used in the original notebook, so output quality may be limited. A translation-specialized multilingual model would be a better choice for a production translator.
