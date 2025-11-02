async function postForm(url, formData) {
	const res = await fetch(url, { method: 'POST', body: formData });
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

async function postJSON(url, obj) {
	const res = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(obj || {})
	});
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

function setOutput(id, data) {
	document.getElementById(id).textContent = JSON.stringify(data, null, 2);
}

// Index dataset
const btnIndex = document.getElementById('btn-index');
btnIndex.addEventListener('click', async () => {
	btnIndex.disabled = true;
	setOutput('index-result', { status: 'Đang xây dựng chỉ mục...' });
	try {
		const root = document.getElementById('index-root').value.trim();
		const resp = await postJSON('/api/index', root ? { root_dir: root } : {});
		
		if (resp.error) {
			setOutput('index-result', { 
				error: resp.error,
				help: 'Vui lòng kiểm tra thư mục dataset và thử lại'
			});
		} else {
			const result = {
				status: 'success',
				message: `Đã xử lý thành công ${resp.indexed}/${resp.count} file PDF`,
				details: resp
			};
			
			if (resp.failed && resp.failed > 0) {
				result.warning = `Có ${resp.failed} file không đọc được`;
				if (resp.failed_files) {
					result.failed_files = resp.failed_files;
				}
			}
			
			setOutput('index-result', result);
			
			// Thêm thông báo thành công
			if (resp.indexed > 0) {
				const successMsg = document.createElement('div');
				successMsg.style.color = 'green';
				successMsg.textContent = '✓ Đã tạo index thành công! Bây giờ bạn có thể tìm kiếm CV tương tự.';
				document.getElementById('index-result').parentNode.appendChild(successMsg);
			}
		}
	} catch (e) {
		let errorMsg = String(e);
		try {
			const errorObj = JSON.parse(errorMsg);
			errorMsg = errorObj.error || errorMsg;
		} catch {
			// Keep original error message if not JSON
		}
		setOutput('index-result', { 
			error: 'Lỗi khi tạo index: ' + errorMsg,
			help: 'Vui lòng kiểm tra:',
			checks: [
				'Thư mục dataset tồn tại và có quyền truy cập',
				'Có file PDF trong thư mục',
				'File PDF không bị lỗi và có thể đọc được'
			]
		});
	} finally {
		btnIndex.disabled = false;
	}
});

// Extract NER
const btnExtractFile = document.getElementById('btn-extract-file');
btnExtractFile.addEventListener('click', async () => {
	const file = document.getElementById('file-extract').files[0];
	if (!file) {
		setOutput('extract-result', { error: 'Vui lòng chọn file PDF' });
		return;
	}
	
	if (!file.name.toLowerCase().endsWith('.pdf')) {
		setOutput('extract-result', { error: 'Chỉ hỗ trợ file PDF' });
		return;
	}
	
	btnExtractFile.disabled = true;
	setOutput('extract-result', { status: 'Đang trích xuất thông tin...' });
	
	const fd = new FormData();
	fd.append('file', file);
	
	try {
		const resp = await postForm('/api/extract', fd);
		if (resp.error) {
			setOutput('extract-result', { error: resp.error });
		} else if (resp.entities && resp.entities.length === 0) {
			setOutput('extract-result', { 
				message: 'Không tìm thấy thông tin trong file PDF này',
				text_length: resp.text_length 
			});
		} else {
			setOutput('extract-result', resp);
		}
	} catch (e) {
		let errorMsg = String(e);
		try {
			const errorObj = JSON.parse(errorMsg);
			errorMsg = errorObj.error || errorMsg;
		} catch {
			// Keep original error message if not JSON
		}
		setOutput('extract-result', { 
			error: 'Lỗi khi trích xuất: ' + errorMsg 
		});
	} finally {
		btnExtractFile.disabled = false;
	}
});

const btnExtractText = document.getElementById('btn-extract-text');
btnExtractText.addEventListener('click', async () => {
	const text = document.getElementById('text-extract').value.trim();
	if (!text) {
		setOutput('extract-result', { error: 'Vui lòng nhập văn bản cần trích xuất' });
		return;
	}
	
	btnExtractText.disabled = true;
	setOutput('extract-result', { status: 'Đang trích xuất thông tin...' });
	
	const fd = new FormData();
	fd.append('text', text);
	
	try {
		const resp = await postForm('/api/extract', fd);
		if (resp.error) {
			setOutput('extract-result', { error: resp.error });
		} else if (resp.entities && resp.entities.length === 0) {
			setOutput('extract-result', { 
				message: 'Không tìm thấy thông tin trong văn bản này',
				text_length: resp.text_length 
			});
		} else {
			setOutput('extract-result', resp);
		}
	} catch (e) {
		let errorMsg = String(e);
		try {
			const errorObj = JSON.parse(errorMsg);
			errorMsg = errorObj.error || errorMsg;
		} catch {
			// Keep original error message if not JSON
		}
		setOutput('extract-result', { 
			error: 'Lỗi khi trích xuất: ' + errorMsg 
		});
	} finally {
		btnExtractText.disabled = false;
	}
});

// Recommend
const btnRecoFile = document.getElementById('btn-reco-file');
btnRecoFile.addEventListener('click', async () => {
	const file = document.getElementById('file-reco').files[0];
	if (!file) {
		setOutput('reco-result', { error: 'Vui lòng chọn file PDF' });
		return;
	}
	if (!file.name.toLowerCase().endsWith('.pdf')) {
		setOutput('reco-result', { error: 'Chỉ hỗ trợ file PDF' });
		return;
	}
	
	btnRecoFile.disabled = true;
	setOutput('reco-result', { status: 'Đang xử lý...' });
	
	const fd = new FormData();
	fd.append('file', file);
	fd.append('k', document.getElementById('topk').value || '5');
	
	try {
		const resp = await postForm('/api/recommend', fd);
		if (resp.error) {
			setOutput('reco-result', { error: resp.error });
		} else if (resp.results && resp.results.length === 0) {
			setOutput('reco-result', { message: 'Không tìm thấy CV tương tự' });
		} else {
			setOutput('reco-result', resp);
		}
	} catch (e) {
		let errorMsg = String(e);
		try {
			const errorObj = JSON.parse(errorMsg);
			errorMsg = errorObj.error || errorMsg;
		} catch {
			// Keep original error message if not JSON
		}
		setOutput('reco-result', { 
			error: 'Lỗi khi xử lý yêu cầu: ' + errorMsg 
		});
	} finally {
		btnRecoFile.disabled = false;
	}
});

const btnRecoText = document.getElementById('btn-reco-text');
btnRecoText.addEventListener('click', async () => {
	const text = document.getElementById('text-reco').value.trim();
	if (!text) {
		setOutput('reco-result', { error: 'Vui lòng nhập nội dung cần tìm kiếm' });
		return;
	}
	
	btnRecoText.disabled = true;
	setOutput('reco-result', { status: 'Đang xử lý...' });
	
	const fd = new FormData();
	fd.append('text', text);
	fd.append('k', document.getElementById('topk').value || '5');
	
	try {
		const resp = await postForm('/api/recommend', fd);
		if (resp.error) {
			setOutput('reco-result', { error: resp.error });
		} else if (resp.results && resp.results.length === 0) {
			setOutput('reco-result', { message: 'Không tìm thấy CV tương tự' });
		} else {
			setOutput('reco-result', resp);
		}
	} catch (e) {
		let errorMsg = String(e);
		try {
			const errorObj = JSON.parse(errorMsg);
			errorMsg = errorObj.error || errorMsg;
		} catch {
			// Keep original error message if not JSON
		}
		setOutput('reco-result', { 
			error: 'Lỗi khi xử lý yêu cầu: ' + errorMsg 
		});
	} finally {
		btnRecoText.disabled = false;
	}
});
