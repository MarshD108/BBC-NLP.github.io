from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from .config import AppConfig


@lru_cache(maxsize=1)
def get_sentence_model() -> SentenceTransformer:
	# Auto-select device for faster encoding when GPU is available
	try:
		import torch
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	except Exception:
		device = 'cpu'
	return SentenceTransformer(AppConfig.SENTENCE_MODEL, device=device)


@lru_cache(maxsize=1)
def get_ner_pipeline():
	tokenizer = AutoTokenizer.from_pretrained(AppConfig.NER_MODEL)
	model = AutoModelForTokenClassification.from_pretrained(AppConfig.NER_MODEL)
	return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def embed_texts(texts: List[str]) -> np.ndarray:
	model = get_sentence_model()
	embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
	return embeddings.astype(np.float32)


def extract_entities(text: str) -> List[Dict]:
	ner = get_ner_pipeline()
	results = ner(text)
	entities: List[Dict] = []
	for r in results:
		entities.append({
			"text": r.get("word") or r.get("entity_group"),
			"label": r.get("entity_group"),
			"start": int(r.get("start", 0)),
			"end": int(r.get("end", 0)),
			"score": float(r.get("score", 0.0)),
		})
	return entities
