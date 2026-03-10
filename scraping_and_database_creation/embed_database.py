import os
import re
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat

def chunk_text(text, max_tokens=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap

    return chunks

def chunk_text_v2(text, max_tokens=500, overlap=50):
    sentences = text.split("\n")
    nr_of_sentences = len(sentences)
    buffer = []
    chunks = []

    for i in range(nr_of_sentences):
        if len(sentences[i].split()) > max_tokens:
            raise Exception("Sentence too long at: " + str(i))
        elif len(sentences[i].split()) + len(buffer) > max_tokens:
            chunks.append(" ".join(flatten(buffer)))
            buffer = flatten(buffer[-overlap:])
            if len(sentences[i].split()) + len(buffer) < max_tokens:
                buffer.append(sentences[i].split())
                buffer = flatten(buffer)
            else:
                raise Exception("Sentence + overlap too long at: " + str(i))
        else:
            buffer.append(sentences[i].split())
            buffer = flatten(buffer)

    if buffer:
        chunks.append(" ".join(flatten(buffer)))

    return chunks

base_folder = os.path.join(os.getcwd(),'data/title17_chapters_v2')
db = []
index = 0

for chapter_folder in Path(base_folder).iterdir():
    if chapter_folder.is_dir():
        for file in chapter_folder.glob('*.txt'):
            section_text = file.read_text(encoding='utf-8')
            section_name = file.stem

            chunks = chunk_text_v2(section_text)
            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                db.append({
                    "index": index,
                    "text": chunk,
                    "metadata": {
                        "title": "USC Title 17",
                        "chapter": chapter_folder.name,
                        "section": section_name,
                        "chunk": i
                    },
                    "embedding": embedding
                })
                index += 1

os.makedirs("../data/embedded_database_v2", exist_ok=True)
with open("../data/embedded_database_v2/title17.json", "w", encoding="utf-8") as f:
    json.dump(db, f, ensure_ascii=False, indent=2)