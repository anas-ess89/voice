"""
app.py - VoiceID Backend
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys, io, wave, tempfile
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings

# Configuration Supabase - VALEURS DIRECTES (pas de os.environ)
SUPABASE_URL = "https://iwdpfafkcgnfkqcskpal.supabase.co"
SUPABASE_KEY = "sb_publishable_CQeLQ3eOqxM3XO7SQGrsAw_NXufiLFI"

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Dossier frontend
_frontend = os.path.join(BASE_DIR, 'frontend')
if not os.path.isdir(_frontend):
    _frontend = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

app = Flask(__name__, static_folder=_frontend, static_url_path='')
app.secret_key = os.urandom(24)
CORS(app, supports_credentials=True)

# Dossiers temporaires pour Render
UPLOAD_FOLDER = '/tmp/uploads'
MODEL_FOLDER = '/tmp/trained_models'
ALLOWED_EXT = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

_voice_processor = None

# ============================================================================
# IMPORTS SUPABASE
# ============================================================================

from supabase_client import supabase, get_current_user, sign_up, login, logout
from supabase_client import save_audio_to_cloud as save_audio_to_supabase
from supabase_client import save_embedding_to_cloud as save_embedding_to_supabase
from supabase_client import save_voice_model_to_cloud
from supabase_client import load_user_voice_models

# ============================================================================
# HELPERS
# ============================================================================

def get_processor():
    global _voice_processor
    if _voice_processor is None:
        from voice_processor import VoiceProcessor
        _voice_processor = VoiceProcessor()
    return _voice_processor

def allowed_file(fname: str) -> bool:
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def ts() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def require_auth():
    user = get_current_user()
    if not user:
        return None
    return user

def convert_to_wav(raw_bytes: bytes, target_sr: int = 16000) -> bytes:
    suffix = '.webm'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(tmp_path)
            seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
            if seg.dBFS != float('-inf'):
                seg = seg.apply_gain(-20.0 - seg.dBFS)
            buf = io.BytesIO()
            seg.export(buf, format='wav')
            return buf.getvalue()
        except Exception:
            import librosa
            wav_arr, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
            return _arr_to_wav(wav_arr, target_sr)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def _arr_to_wav(wav_arr: np.ndarray, sr: int = 16000) -> bytes:
    rms = np.sqrt(np.mean(wav_arr ** 2))
    if rms > 0:
        target_rms = 10 ** (-30 / 20)
        wav_arr = wav_arr * (target_rms / rms)
    wav_arr = np.clip(wav_arr, -1.0, 1.0)
    pcm = (wav_arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def serve_index():
    user = get_current_user()
    if not user:
        return send_from_directory(app.static_folder, 'auth.html')
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health')
def health():
    vp = get_processor()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(vp.speaker_models),
        'device': vp.device,
    })

@app.route('/api/speakers', methods=['GET'])
def list_speakers():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    models = load_user_voice_models(user['id'])
    speakers = [{'name': name, 'n_samples': model.get('n_samples', 0)} for name, model in models.items()]
    return jsonify({'speakers': speakers})

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print("=" * 55)
    print("  VoiceID — Neural Voice Recognition with Supabase")
    print("=" * 55)
    print(f"  Listening on port {port}")
    print("=" * 55)
    app.run(debug=False, host='0.0.0.0', port=port)
