import io
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from .services import (
	extract_entities_from_text,
	index_dataset_pdfs,
	recommend_similar_candidates,
	read_pdf_file_to_text,
)


main_bp = Blueprint("main", __name__)
api_bp = Blueprint("api", __name__)


@main_bp.route("/")
def index():
	return render_template("index.html")


@api_bp.route("/extract", methods=["POST"])
def api_extract() -> Any:
	try:
		text: str = ""
		
		if "file" in request.files:
			file = request.files["file"]
			if not file or file.filename == "":
				return jsonify({"error": "Vui lòng chọn file"}), 400
				
			if not file.filename.lower().endswith('.pdf'):
				return jsonify({"error": "Chỉ hỗ trợ file PDF"}), 400
				
			try:
				file_bytes = file.read()
				text = read_pdf_file_to_text(io.BytesIO(file_bytes))
			except Exception as e:
				return jsonify({"error": f"Lỗi đọc file PDF: {str(e)}"}), 400
		else:
			text = request.form.get("text", "").strip()

		if not text:
			return jsonify({"error": "Không tìm thấy nội dung text để xử lý"}), 400

		entities = extract_entities_from_text(text)
		
		if not entities:
			return jsonify({
				"entities": [],
				"message": "Không tìm thấy thực thể nào trong văn bản",
				"text_length": len(text)
			})
			
		return jsonify({
			"entities": entities,
			"count": len(entities),
			"text_length": len(text)
		})

	except Exception as e:
		return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


@api_bp.route("/index", methods=["POST"])
def api_index() -> Any:
	root_dir = request.json.get("root_dir") if request.is_json else None
	stats = index_dataset_pdfs(root_dir_override=root_dir)
	return jsonify(stats)


@api_bp.route("/recommend", methods=["POST"])
def api_recommend() -> Any:
	try:
		k = min(max(int(request.form.get("k", 5)), 1), 50)  # Limit k between 1 and 50
		text: str = ""

		if "file" in request.files:
			file = request.files["file"]
			if not file or file.filename == "":
				return jsonify({"error": "No file selected"}), 400
			if not file.filename.lower().endswith('.pdf'):
				return jsonify({"error": "Only PDF files are supported"}), 400
			try:
				text = read_pdf_file_to_text(file.stream)
			except Exception as e:
				return jsonify({"error": f"Failed to read PDF file: {str(e)}"}), 400
		else:
			text = request.form.get("text", "").strip()

		if not text:
			return jsonify({"error": "No text content found"}), 400

		results = recommend_similar_candidates(text, top_k=k)
		if not results:
			return jsonify({"results": [], "message": "No similar candidates found"}), 200

		return jsonify({
			"results": results,
			"count": len(results),
			"query_length": len(text)
		})

	except ValueError as ve:
		return jsonify({"error": str(ve)}), 400
	except Exception as e:
		return jsonify({"error": f"Internal server error: {str(e)}"}), 500
