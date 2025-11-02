from io import BytesIO
from typing import Optional

from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader


def read_pdf_file_to_text(stream_or_bytes) -> str:
	try:
		if isinstance(stream_or_bytes, (bytes, bytearray)):
			data = BytesIO(stream_or_bytes)
		else:
			data = stream_or_bytes
		text = extract_text(data)
		if text and text.strip():
			return text
	except Exception:
		pass

	try:
		if isinstance(stream_or_bytes, (bytes, bytearray)):
			data = BytesIO(stream_or_bytes)
		else:
			data = stream_or_bytes
		reader = PdfReader(data)
		pages = [p.extract_text() or "" for p in reader.pages]
		return "\n".join(pages)
	except Exception:
		return ""
