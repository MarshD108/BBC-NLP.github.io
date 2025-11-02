<<<<<<< HEAD
# CV Intelligence (BERT NER + Candidate Recommendation)

Ứng dụng web Flask trích xuất thực thể từ CV tiếng Anh bằng BERT NER và gợi ý ứng viên phù hợp cho doanh nghiệp bằng Sentence-BERT.

## Yêu cầu
- Python 3.10+
- Windows PowerShell

## Cài đặt
```powershell
cd "D:\NLP Project báo cáo cuối kì"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Lần đầu chạy sẽ tải model từ Hugging Face (~100-500MB).

## Chạy server
```powershell
$env:FLASK_ENV="development"
python run.py
```
Mở trình duyệt: http://localhost:5000

## Dữ liệu
- Dataset mặc định ở `Dataset/data/data/` theo cấu trúc thư mục ngành/nghề và file PDF.
- Bạn có thể đặt ENV `DATASET_DIR` để chỉ định thư mục khác.

## Chỉ mục hoá dữ liệu
- Vào trang chủ, mục "Xây dựng chỉ mục" và bấm nút.
- Hoặc gọi API:
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5000/api/index -ContentType 'application/json' -Body '{}'
```

## API
- POST `/api/extract` (form-data: `file` PDF hoặc `text`): trả về `entities`.
- POST `/api/recommend` (form-data: `file` hoặc `text`, `k`): trả về `results` (path, role, score).
- POST `/api/index` (json: `{ "root_dir": "optional" }`): dựng chỉ mục từ dataset.

## Ghi chú kỹ thuật
- NER: `dslim/bert-base-NER` (aggregation simple)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (chuẩn hoá L2)
- Tìm kiếm: cosine similarity top-k
- Lưu index: `data_index/embeddings.npy`, `data_index/index.jsonl`, `data_index/texts.jsonl`

## Bảo trì
- Nếu thay đổi DATASET, chạy lại "Xây dựng chỉ mục".
- Có thể đổi model qua ENV `NER_MODEL`, `SENTENCE_MODEL`.
=======
# BBC-NLP.github.io
>>>>>>> e8cb2bcd306a2c7f8685cdd03f46b52cefb3a6bd
