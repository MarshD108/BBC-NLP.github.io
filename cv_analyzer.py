#!/usr/bin/env python3
"""
CV Intelligence - BERT NER & Candidate Recommendation
Phiên bản đơn giản không cần Flask, chạy trực tiếp từ command line
"""
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Kiểm tra dataset
def check_dataset():
    dataset_dir = Path("Dataset/data/data")
    if not dataset_dir.exists():
        print("❌ Không tìm thấy dataset tại:", dataset_dir)
        return False
    
    categories = [d for d in dataset_dir.iterdir() if d.is_dir()]
    print(f"✅ Tìm thấy {len(categories)} danh mục nghề nghiệp:")
    
    total_pdfs = 0
    for cat in categories[:10]:  # Hiển thị 10 danh mục đầu
        pdfs = list(cat.glob("*.pdf"))
        total_pdfs += len(pdfs)
        print(f"  📁 {cat.name}: {len(pdfs)} CVs")
    
    print(f"📊 Tổng cộng: {total_pdfs} CVs")
    return True

# Trích xuất text từ PDF (cần cài pdfminer)
def extract_pdf_text(pdf_path: str) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except ImportError:
        print("❌ Cần cài pdfminer: pip install pdfminer.six")
        return ""
    except Exception as e:
        print(f"❌ Lỗi đọc PDF {pdf_path}: {e}")
        return ""

# BERT NER (cần cài transformers)
def extract_entities(text: str) -> List[Dict[str, Any]]:
    try:
        from transformers import pipeline
        ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        results = ner(text)
        
        entities = []
        for r in results:
            entities.append({
                "text": r.get("word", ""),
                "label": r.get("entity_group", ""),
                "score": float(r.get("score", 0.0))
            })
        return entities
    except ImportError:
        print("❌ Cần cài transformers: pip install transformers torch")
        return []
    except Exception as e:
        print(f"❌ Lỗi NER: {e}")
        return []

# Sentence-BERT embeddings (cần cài sentence-transformers)
def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()
    except ImportError:
        print("❌ Cần cài sentence-transformers: pip install sentence-transformers")
        return []
    except Exception as e:
        print(f"❌ Lỗi embeddings: {e}")
        return []

# Tìm CV tương tự
def find_similar_cvs(query_text: str, dataset_dir: str, top_k: int = 5) -> List[Dict[str, Any]]:
    print(f"🔍 Tìm kiếm CV tương tự với: '{query_text[:50]}...'")
    
    # Lấy danh sách PDFs
    pdfs = []
    for cat_dir in Path(dataset_dir).iterdir():
        if cat_dir.is_dir():
            for pdf in cat_dir.glob("*.pdf"):
                pdfs.append((str(pdf), cat_dir.name))
    
    if not pdfs:
        print("❌ Không tìm thấy PDF nào")
        return []
    
    print(f"📚 Đang xử lý {len(pdfs)} CVs...")
    
    # Trích xuất text từ một số PDF mẫu (để test)
    sample_pdfs = pdfs[:10]  # Chỉ lấy 10 PDF đầu để test
    texts = []
    valid_pdfs = []
    
    for pdf_path, category in sample_pdfs:
        text = extract_pdf_text(pdf_path)
        if text.strip():
            texts.append(text)
            valid_pdfs.append((pdf_path, category))
    
    if not texts:
        print("❌ Không trích xuất được text từ PDF nào")
        return []
    
    # Tạo embeddings
    print("🧠 Tạo embeddings...")
    embeddings = get_embeddings([query_text] + texts)
    
    if not embeddings:
        print("❌ Không tạo được embeddings")
        return []
    
    # Tính similarity (cosine)
    query_emb = embeddings[0]
    results = []
    
    for i, (pdf_path, category) in enumerate(valid_pdfs):
        doc_emb = embeddings[i + 1]
        # Cosine similarity = dot product (vì đã normalize)
        similarity = sum(a * b for a, b in zip(query_emb, doc_emb))
        results.append({
            "path": pdf_path,
            "category": category,
            "similarity": similarity
        })
    
    # Sắp xếp theo similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def main():
    print("🚀 CV Intelligence - BERT NER & Candidate Recommendation")
    print("=" * 60)
    
    # Kiểm tra dataset
    if not check_dataset():
        return
    
    print("\n" + "=" * 60)
    print("📋 MENU:")
    print("1. Kiểm tra dataset")
    print("2. Trích xuất thực thể từ text")
    print("3. Tìm CV tương tự")
    print("4. Thoát")
    
    while True:
        try:
            choice = input("\nChọn chức năng (1-4): ").strip()
            
            if choice == "1":
                check_dataset()
                
            elif choice == "2":
                text = input("Nhập text CV (hoặc Enter để dùng text mẫu): ").strip()
                if not text:
                    text = "John Smith is a Software Engineer with 5 years of experience in Python and JavaScript. He worked at Google and Microsoft. Contact: john@email.com"
                    print(f"Sử dụng text mẫu: {text[:50]}...")
                
                print("🔍 Đang trích xuất thực thể...")
                entities = extract_entities(text)
                
                if entities:
                    print("✅ Thực thể được trích xuất:")
                    for entity in entities:
                        print(f"  📝 {entity['text']} -> {entity['label']} (độ tin cậy: {entity['score']:.2f})")
                else:
                    print("❌ Không trích xuất được thực thể nào")
                    
            elif choice == "3":
                query = input("Nhập mô tả công việc/CV để tìm kiếm: ").strip()
                if not query:
                    query = "Software Engineer Python JavaScript"
                    print(f"Sử dụng query mẫu: {query}")
                
                results = find_similar_cvs(query, "Dataset/data/data", top_k=5)
                
                if results:
                    print("🎯 CV tương tự nhất:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {result['category']} (độ tương tự: {result['similarity']:.3f})")
                        print(f"     📄 {result['path']}")
                else:
                    print("❌ Không tìm thấy CV tương tự")
                    
            elif choice == "4":
                print("👋 Tạm biệt!")
                break
                
            else:
                print("❌ Lựa chọn không hợp lệ")
                
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
