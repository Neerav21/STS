from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow_hub as hub
#import tensorflow_text  # Necessary for some USE models, even if not directly used
import numpy as np

app = FastAPI()

# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class Texts(BaseModel):
    text1: str
    text2: str

@app.post("/similarity/")
def get_similarity(texts: Texts):
    embeddings = embed([texts.text1, texts.text2])
    similarity = np.inner(embeddings[0], embeddings[1]).item()
    return {"similarity": similarity}

#deployment ahead