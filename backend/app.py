"""
app.py - VoiceID Backend (version de test pour Render)
"""
from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')

@app.route('/')
def index():
    return jsonify({"message": "VoiceID API is running!", "status": "ok"})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "test": True})

# Route pour servir les fichiers frontend
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
