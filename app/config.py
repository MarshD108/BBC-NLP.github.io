import os
from pathlib import Path


class AppConfig:
	SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
	BASE_DIR = Path(__file__).resolve().parent.parent
	DATASET_DIR = os.environ.get(
		"DATASET_DIR",
		str(Path(BASE_DIR, "Dataset", "data", "data")),
	)
	INDEX_DIR = os.environ.get("INDEX_DIR", str(Path(BASE_DIR, "data_index")))
	LOG_DIR = os.environ.get("LOG_DIR", str(Path(BASE_DIR, "logs")))
	MAX_CONTENT_LENGTH = 25 * 1024 * 1024

	SENTENCE_MODEL = os.environ.get("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
	NER_MODEL = os.environ.get("NER_MODEL", "dslim/bert-base-NER")
