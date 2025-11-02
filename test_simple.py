#!/usr/bin/env python3
"""
Test script to verify Flask app works without heavy ML dependencies
"""
from flask import Flask, jsonify, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head><title>CV Intelligence - Test</title></head>
<body>
    <h1>CV Intelligence - Test Mode</h1>
    <p>Flask app is running! This is a simplified version for testing.</p>
    <p>Dataset directory: {{ dataset_dir }}</p>
    <p>PDF files found: {{ pdf_count }}</p>
</body>
</html>
    ''', dataset_dir=os.environ.get('DATASET_DIR', 'Dataset/data/data'), pdf_count=0)

@app.route('/api/test')
def api_test():
    return jsonify({"status": "ok", "message": "API is working"})

if __name__ == '__main__':
    print("Starting test server...")
    print("Open: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
