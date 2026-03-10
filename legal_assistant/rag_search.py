import numpy as np
import json
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Number of top chunks to return in searches
NUM_OF_TOP = 10


def load_json(file_path: str) -> List[Dict]:
    """
    Load the embedded chunk database from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list[dict]: List of chunks, each containing 'text', 'embedding', 'index', 'metadata'.
    """
    with open(file_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array between 0 and 1.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array.
    """
    arr = np.array(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else np.zeros_like(arr)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_search(chunks: List[Dict], query: str, model: SentenceTransformer) -> List[int]:
    """
    Retrieve top chunks based on cosine similarity between query and chunk embeddings.

    Returns:
        list[int]: Indices of top chunks.
    """
    query_embedding = model.encode(query)
    top_scores = np.full(NUM_OF_TOP, -np.inf)
    top_indices = np.full(NUM_OF_TOP, -1, dtype=int)

    for chunk in chunks:
        similarity = cosine_similarity(chunk["embedding"], query_embedding)
        if similarity > top_scores.min():
            min_index = np.argmin(top_scores)
            top_scores[min_index] = similarity
            top_indices[min_index] = chunk["index"]

    sorted_pairs = sorted(zip(top_scores, top_indices), reverse=True)
    _, sorted_indices = zip(*sorted_pairs)

    return list(sorted_indices)


def build_bm25_index(chunks: List[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    """
    Build a BM25 index from text chunks.

    Returns:
        (BM25Okapi, tokenized_corpus)
    """
    tokenized_corpus = [word_tokenize(chunk["text"].lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def bm25_search(chunks: List[Dict], query: str) -> List[int]:
    """
    Perform BM25 search and return top chunk indices.
    """
    bm25, _ = build_bm25_index(chunks)
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [int(chunks[i]["index"]) for i in ranked_indices[:NUM_OF_TOP]]


def two_stage_retrieval(chunks: List[Dict], query: str, model: SentenceTransformer) -> List[int]:
    """
    First retrieve top chunks with BM25, then re-rank with cosine similarity.
    """
    bm25_results = bm25_search(chunks, query)
    bm25_top_chunks = [chunk for chunk in chunks if chunk["index"] in bm25_results]
    return cosine_search(bm25_top_chunks, query, model)


def hybrid_search(
    chunks: List[Dict],
    query: str,
    model: SentenceTransformer,
    alpha: float = 0.7
) -> Dict[str, str]:
    """
    Hybrid retrieval combining BM25 and cosine similarity scores.

    Args:
        chunks: List of embedded chunks.
        query: User query string.
        model: SentenceTransformer model.
        alpha: Weight between BM25 (alpha) and cosine (1 - alpha).

    Returns:
        dict[str, str]: Mapping of section name -> chunk text for top results.
    """
    bm25, _ = build_bm25_index(chunks)
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = np.array(min_max_normalize(bm25.get_scores(tokenized_query)))

    query_embedding = model.encode(query)
    cosine_scores = np.array([
        cosine_similarity(chunk["embedding"], query_embedding)
        for chunk in chunks
    ])
    cosine_scores = min_max_normalize(cosine_scores)

    # Hybrid score
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * cosine_scores
    ranked_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)

    results = {}
    for chunk in chunks:
        if chunk["index"] in ranked_indices[:NUM_OF_TOP]:
            section = chunk["metadata"].get("section", f"Index {chunk['index']}")
            results[section] = chunk["text"]

    return results
