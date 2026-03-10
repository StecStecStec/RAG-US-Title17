import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from .rag_search import load_json, hybrid_search
from pathlib import Path


load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Check your .env file.")

client = Groq(api_key=api_key)
script_dir = Path(__file__).parent
_DB_PATH = script_dir.parent / "data" / "embedded_database_v2" / "title17.json"
_DB_PATH = str(_DB_PATH)
_sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
_chunks = load_json(_DB_PATH)

_SYSTEM_PROMPT = (
    "You are an expert in U.S. copyright law. "
    "You provide legally-informed but educational explanations, "
    "breaking down statutes, case law, and fair use principles clearly. "
    "You do not give legal advice, but you summarize laws and explain their application in plain English."
)

def ask_lawyer(query: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
    relevant_info = hybrid_search(_chunks, query, _sentence_model)
    context = "\n".join([f"Section: {k}\nContent: {v}" for k, v in relevant_info.items()])

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{query}\n\nRelevant retrieved documents:\n{context}"}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content
