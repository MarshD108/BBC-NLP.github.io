import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import multiprocessing

import numpy as np
from tqdm import tqdm

# Số luồng xử lý tối đa cho PDF processing
MAX_WORKERS = max(4, multiprocessing.cpu_count() - 1)  # Để lại 1 core cho hệ thống
CHUNK_SIZE = 50  # Số file xử lý mỗi chunk

from .config import AppConfig
from .nlp import embed_texts, extract_entities
from .pdf_utils import read_pdf_file_to_text


@dataclass
class IndexItem:
    path: str
    role: str
    text_path: str


class SearchCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        if query in self.cache:
            self.access_times[query] = time.time()
            return self.cache[query]
        return None

    def set(self, query: str, results: List[Dict[str, Any]]) -> None:
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_query = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_query]
            del self.access_times[oldest_query]
        self.cache[query] = results
        self.access_times[query] = time.time()


BATCH_SIZE = 32  # Kích thước batch cho xử lý
search_cache = SearchCache()  # Cache cho kết quả tìm kiếm

INDEX_META = Path(AppConfig.INDEX_DIR) / "index.jsonl"
EMBED_NPY = Path(AppConfig.INDEX_DIR) / "embeddings.npy"
TEXTS_NPY = Path(AppConfig.INDEX_DIR) / "texts.jsonl"
FAISS_INDEX = Path(AppConfig.INDEX_DIR) / "index.faiss"
HNSW_INDEX = Path(AppConfig.INDEX_DIR) / "index.hnsw"


def ensure_dirs() -> None:
	Path(AppConfig.INDEX_DIR).mkdir(parents=True, exist_ok=True)
	Path(AppConfig.LOG_DIR).mkdir(parents=True, exist_ok=True)


def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
	return extract_entities(text)


def _iter_pdf_files(root_dir: Optional[str]) -> List[Tuple[str, str]]:
    root = Path(root_dir) if root_dir else Path(AppConfig.DATASET_DIR)
    pairs: List[Tuple[str, str]] = []
    if not root.exists():
        return pairs
    for role_dir in root.iterdir():
        if role_dir.is_dir():
            role = role_dir.name
            for pdf in role_dir.glob("*.pdf"):
                pairs.append((str(pdf), role))
    return pairs

def _process_single_pdf(pdf_info: Tuple[str, str]) -> Tuple[Optional[str], Optional[IndexItem], Optional[str]]:
    """Xử lý một file PDF đơn lẻ"""
    pdf_path, role = pdf_info
    try:
        with open(pdf_path, "rb") as f:
            text = read_pdf_file_to_text(f)
            if text.strip():
                return text, IndexItem(path=pdf_path, role=role, text_path=""), None
            return None, None, pdf_path
    except Exception as e:
        print(f"Lỗi đọc file {pdf_path}: {str(e)}")
        return None, None, pdf_path

def _process_pdf_chunk(chunk: List[Tuple[str, str]]) -> Tuple[List[str], List[IndexItem], List[str]]:
    """Xử lý một chunk các file PDF song song"""
    texts: List[str] = []
    items: List[IndexItem] = []
    failed_files: List[str] = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_process_single_pdf, pdf_info) for pdf_info in chunk]
        for future in as_completed(futures):
            text, item, failed = future.result()
            if text and item:
                texts.append(text)
                items.append(item)
            elif failed:
                failed_files.append(failed)
    
    return texts, items, failed_files

def _build_hnsw_index(embeddings: np.ndarray) -> Optional[str]:
    """Tạo HNSW index cho tìm kiếm nhanh"""
    try:
        import hnswlib
        dim = embeddings.shape[1]
        num_elements = embeddings.shape[0]
        
        # Khởi tạo index
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # Thêm vectors vào index
        index.add_items(embeddings)
        
        # Lưu index
        index.save_index(str(HNSW_INDEX))
        return str(HNSW_INDEX)
    except Exception as e:
        print(f"Không thể tạo HNSW index: {str(e)}")
        return None


def _search_hnsw(query_emb: np.ndarray, k: int) -> np.ndarray:
    """Tìm kiếm sử dụng HNSW index"""
    try:
        import hnswlib
        # Load index
        dim = query_emb.shape[0]
        p = hnswlib.Index(space='cosine', dim=dim)
        p.load_index(str(HNSW_INDEX))
        
        # Search
        labels, _ = p.knn_query(query_emb.reshape(1, -1), k=k)
        return labels[0]
    except Exception as e:
        raise ValueError(f"HNSW search failed: {str(e)}")

def index_dataset_pdfs(root_dir_override: Optional[str] = None) -> Dict[str, Any]:
    try:
        ensure_dirs()
        root_dir = root_dir_override if root_dir_override else AppConfig.DATASET_DIR
        
        if not Path(root_dir).exists():
            raise ValueError(f"Thư mục dataset không tồn tại: {root_dir}")
            
        pdfs = _iter_pdf_files(root_dir_override)
        if not pdfs:
            raise ValueError(f"Không tìm thấy file PDF nào trong thư mục: {root_dir}")
            
        print(f"Đang xử lý {len(pdfs)} file PDF...")
        
        texts: List[str] = []
        items: List[IndexItem] = []
        failed_files: List[str] = []

        # Xử lý PDF theo chunks với progress bar
        chunks = [pdfs[i:i + CHUNK_SIZE] for i in range(0, len(pdfs), CHUNK_SIZE)]
        with tqdm(total=len(pdfs), desc="Đang xử lý PDF") as pbar:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(_process_pdf_chunk, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    chunk_texts, chunk_items, chunk_failed = future.result()
                    texts.extend(chunk_texts)
                    items.extend(chunk_items)
                    failed_files.extend(chunk_failed)
                    pbar.update(CHUNK_SIZE)

        if not texts:
            raise ValueError("Không thể trích xuất nội dung từ bất kỳ file PDF nào")

        print("Đang tạo embeddings...")
        # Tạo embeddings theo batch để tránh OOM
        embeddings_list = []
        with tqdm(total=len(texts), desc="Tạo embeddings") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_emb = embed_texts(batch_texts)
                embeddings_list.append(batch_emb)
                pbar.update(len(batch_texts))
        
        emb = np.vstack(embeddings_list)
        if emb.shape[0] == 0:
            raise ValueError("Không thể tạo embeddings từ văn bản")
            
        print("Đang lưu index...")
        np.save(EMBED_NPY, emb)

        # Try to build HNSW index first (faster and more Windows-friendly)
        hnsw_path = None
        try:
            hnsw_path = _build_hnsw_index(emb)
        except Exception as e:
            print(f"Không thể tạo HNSW index: {str(e)}")
            hnsw_path = None

        # Try FAISS as fallback
        faiss_path = None
        if not hnsw_path:
            try:
                faiss_path = _build_faiss_index(emb)
            except Exception:
                faiss_path = None

        with open(INDEX_META, "w", encoding="utf-8") as w:
            for item in items:
                w.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

        with open(TEXTS_NPY, "w", encoding="utf-8") as w:
            for t in texts:
                w.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

        result = {
            "status": "success",
            "count": len(pdfs),
            "indexed": len(texts),
            "failed": len(failed_files),
            "failed_files": failed_files if failed_files else None,
            "embedding_shape": list(emb.shape),
            "hnsw_index": hnsw_path,
            "faiss_index": faiss_path
        }
        
        print(f"Hoàn thành! Đã index {len(texts)}/{len(pdfs)} file PDF")
        return result
        
    except Exception as e:
        print(f"Lỗi khi tạo index: {str(e)}")
        raise ValueError(f"Lỗi khi tạo index: {str(e)}")


def _load_index() -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
	if not EMBED_NPY.exists() or not INDEX_META.exists():
		return np.zeros((0, 384), dtype=np.float32), [], []
	emb = np.load(EMBED_NPY)
	meta: List[Dict[str, Any]] = []
	with open(INDEX_META, "r", encoding="utf-8") as f:
		for line in f:
			meta.append(json.loads(line))
	texts: List[str] = []
	if TEXTS_NPY.exists():
		with open(TEXTS_NPY, "r", encoding="utf-8") as f:
			for line in f:
				obj = json.loads(line)
				texts.append(obj.get("text", ""))
	return emb, meta, texts


def _build_faiss_index(emb: np.ndarray) -> Optional[str]:
	"""Try to build a FAISS index (if faiss is installed). Returns path or None."""
	try:
		import faiss
	except Exception:
		return None

	# emb should be float32 and L2-normalized (sentence-transformers can normalize)
	if emb.dtype != np.float32:
		emb = emb.astype(np.float32)

	d = emb.shape[1]
	# Use inner-product on normalized vectors for cosine similarity
	index = faiss.IndexFlatIP(d)
	try:
		index.add(emb)
		faiss.write_index(index, str(FAISS_INDEX))
		return str(FAISS_INDEX)
	except Exception:
		return None


def _load_faiss_index() -> Optional[object]:
	try:
		import faiss
	except Exception:
		return None
	if not FAISS_INDEX.exists():
		return None
	try:
		idx = faiss.read_index(str(FAISS_INDEX))
		return idx
	except Exception:
		return None


def _build_hnsw_index(emb: np.ndarray, space: str = 'cosine') -> Optional[str]:
	"""Build a HNSWLib index and save to disk. Returns path or None."""
	try:
		import hnswlib
	except Exception:
		return None

	# Ensure float32
	if emb.dtype != np.float32:
		emb = emb.astype(np.float32)

	num_elements, dim = emb.shape
	try:
		p = hnswlib.Index(space=space, dim=dim)
		# set M/P ef parameters to sensible defaults
		p.init_index(max_elements=num_elements, ef_construction=200, M=16)
		p.add_items(emb, list(range(num_elements)))
		p.set_ef(50)
		p.save_index(str(HNSW_INDEX))
		return str(HNSW_INDEX)
	except Exception:
		return None


def _load_hnsw_index() -> Optional[object]:
	try:
		import hnswlib
	except Exception:
		return None
	if not HNSW_INDEX.exists():
		return None
	try:
		# need to know dim and space: we can infer dim from embeddings.npy
		emb = np.load(EMBED_NPY)
		dim = emb.shape[1]
		p = hnswlib.Index(space='cosine', dim=dim)
		p.load_index(str(HNSW_INDEX))
		p.set_ef(50)
		return p
	except Exception:
		return None


def recommend_similar_candidates(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
	if not query_text or not query_text.strip():
		raise ValueError("Query text cannot be empty")
	
	emb, meta, texts = _load_index()
	if emb.shape[0] == 0:
		raise ValueError("No index found. Please build index first.")
		
	try:
		q = embed_texts([query_text])  # already L2-normalized
		if q.shape[0] == 0 or q.shape[1] != emb.shape[1]:
			raise ValueError("Failed to create embedding for query text")

		# Try HNSW (fast, pip-installable on Windows) first, then FAISS if available
		hnsw_idx = _load_hnsw_index()
		faiss_idx = _load_faiss_index()
		results: List[Dict[str, Any]] = []
		if hnsw_idx is not None:
			try:
				labels, distances = hnsw_idx.knn_query(q.astype(np.float32), k=max(top_k, 1))
				for i, idx in enumerate(labels[0]):
					if idx < 0:
						continue
					score = float(1.0 - distances[0][i]) if distances[0][i] >= 0 else 0.0
					if score <= 0:
						continue
					item = meta[int(idx)]
					results.append({
						"path": item.get("path"),
						"role": item.get("role"),
						"score": score,
						"text": texts[int(idx)] if texts else ""
					})
				return results
			except Exception:
				pass
		
		if faiss_idx is not None:
			try:
				# FAISS expects float32
				q_f = q.astype(np.float32)
				D, I = faiss_idx.search(q_f, max(top_k, 1))
				for i, idx in enumerate(I[0]):
					if idx < 0:
						continue
					score = float(D[0][i])
					if score <= 0:
						continue
					item = meta[int(idx)]
					results.append({
						"path": item.get("path"),
						"role": item.get("role"),
						"score": score,
						"text": texts[int(idx)] if texts else ""
					})
				return results
			except Exception:
				# Fallthrough to numpy search
				pass

		# cosine similarity = dot product for normalized vectors (fallback)
		scores = (q @ emb.T)[0]
		indices = np.argsort(-scores)[: max(top_k, 1)]
		for idx in indices:
			item = meta[int(idx)]
			score = float(scores[int(idx)])
			if score > 0:  # Only include results with positive similarity
				results.append({
					"path": item.get("path"),
					"role": item.get("role"),
					"score": score,
					"text": texts[int(idx)] if texts else ""
				})
		return results
	except Exception as e:
		raise ValueError(f"Error processing query: {str(e)}")


__all__ = [
	"read_pdf_file_to_text",
	"extract_entities_from_text",
	"index_dataset_pdfs",
	"recommend_similar_candidates",
]
