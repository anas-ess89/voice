"""
app.py - VoiceID Backend avec Supabase Auth et Stockage Cloud
"""
import os

# Configuration pour Render
UPLOAD_FOLDER_RENDER = '/tmp/uploads'
MODEL_FOLDER_RENDER = '/tmp/trained_models'

os.makedirs(UPLOAD_FOLDER_RENDER, exist_ok=True)
os.makedirs(MODEL_FOLDER_RENDER, exist_ok=True)

# Configuration Supabase - valeurs directes
SUPABASE_URL = "https://iwdpfafkcgnfkqcskpal.supabase.co"
SUPABASE_KEY = "sb_publishable_CQeLQ3eOqxM3XO7SQGrsAw_NXufiLFI"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys, io, wave, tempfile
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

_frontend = os.path.join(BASE_DIR, 'frontend')
if not os.path.isdir(_frontend):
    _frontend = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

app = Flask(__name__, static_folder=_frontend, static_url_path='')
app.secret_key = os.urandom(24)
CORS(app, supports_credentials=True)

# Dossiers temporaires pour uploads (Render utilise /tmp)
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
# ROUTES STATIQUES
# ============================================================================

@app.route('/')
def serve_index():
    user = get_current_user()
    if not user:
        return send_from_directory(app.static_folder, 'auth.html')
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path.startswith('api'):
        from flask import abort
        abort(404)
    return send_from_directory(app.static_folder, path)

# ============================================================================
# ROUTES AUTH
# ============================================================================

@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    username = data.get('username')
    
    if not email or not password or not username:
        return jsonify({'error': 'Email, password and username required'}), 400
    
    result = sign_up(email, password, username)
    return jsonify(result)

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    result = login(email, password)
    return jsonify(result)

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    result = logout()
    return jsonify(result)

@app.route('/api/auth/me', methods=['GET'])
def api_me():
    user = get_current_user()
    if user:
        return jsonify({'authenticated': True, 'user': user})
    return jsonify({'authenticated': False, 'user': None})

# ============================================================================
# ROUTES API - UPLOAD
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    file = request.files['audio']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = f"{ts()}_{file.filename}"
    
    audio_bytes = file.read()
    wav_bytes = convert_to_wav(audio_bytes)
    
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
    with open(temp_path, 'wb') as f:
        f.write(wav_bytes)

    try:
        vp = get_processor()
        result = vp.process_audio(temp_path)
        result.update({'filename': filename, 'recorded': False})

        if result.get('embedding_extracted'):
            emb = vp.get_embedding(temp_path)
            if emb is not None:
                save_audio_to_supabase(user['id'], wav_bytes, filename, "train")
                save_embedding_to_supabase(user['id'], filename, emb)
                
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ============================================================================
# ROUTES API - RECORD
# ============================================================================

@app.route('/api/record', methods=['POST'])
def record_audio():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio blob received'}), 400

    blob = request.files['audio']
    raw = blob.read()
    prefix = request.form.get('prefix', 'rec')

    if len(raw) < 500:
        return jsonify({'error': 'Recording too short or empty'}), 400

    try:
        wav_bytes = convert_to_wav(raw)
        filename = f"{prefix}_{ts()}.wav"
        
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
        with open(temp_path, 'wb') as f:
            f.write(wav_bytes)

        vp = get_processor()
        result = vp.process_audio(temp_path)
        result.update({'filename': filename, 'recorded': True})

        if result.get('embedding_extracted'):
            emb = vp.get_embedding(temp_path)
            if emb is not None:
                save_audio_to_supabase(user['id'], wav_bytes, filename, "record")
                save_embedding_to_supabase(user['id'], filename, emb)
                
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ============================================================================
# ROUTES API - TRAIN
# ============================================================================

@app.route('/api/train', methods=['POST'])
def train_model():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    data = request.json or {}
    speaker_name = data.get('speaker_name', '').strip()
    files_to_train = data.get('files', [])
    
    if not speaker_name:
        return jsonify({'error': 'Speaker name required'}), 400
    if not files_to_train:
        return jsonify({'error': 'No files provided'}), 400
    
    try:
        embeddings = []
        successful_files = []
        
        for file_info in files_to_train:
            filename = file_info.get('filename')
            if not filename:
                continue
            
            result = supabase.table("embeddings").select("*").eq("audio_file_name", filename).eq("user_id", user['id']).execute()
            
            if result.data:
                emb_vector = result.data[0]['embedding_vector']
                emb_array = np.array(emb_vector)
                embeddings.append(emb_array)
                successful_files.append(filename)
                print(f"  ✓ {filename} (depuis cloud - user {user['id']})")
            else:
                print(f"  ✗ {filename} (non trouvé pour cet utilisateur)")
        
        if len(embeddings) < 1:
            return jsonify({'error': 'Aucun embedding valide trouvé pour cet utilisateur'}), 400
        
        arr = np.array(embeddings)
        mean_emb = arr.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        
        intra_sims = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                intra_sims.append(float(np.dot(arr[i], arr[j])))
        
        intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
        
        model = {
            'speaker_name': speaker_name,
            'user_id': user['id'],
            'embeddings': [e.tolist() for e in embeddings],
            'mean_embedding': mean_emb.tolist(),
            'n_samples': len(embeddings),
            'training_date': datetime.now().isoformat(),
            'intra_similarity_mean': intra_mean,
            'files': successful_files
        }
        
        save_voice_model_to_cloud(user['id'], speaker_name, model)
        
        vp = get_processor()
        vp.speaker_models[speaker_name] = model
        
        return jsonify({
            'status': 'success',
            'speaker_name': speaker_name,
            'n_samples': len(embeddings),
            'intra_similarity_mean': intra_mean,
            'message': f'Modèle entraîné pour {speaker_name}'
        })
        
    except Exception as e:
        print(f"Erreur train: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ROUTES API - IDENTIFY
# ============================================================================

@app.route('/api/identify', methods=['POST'])
def identify_speaker():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    file = request.files['audio']
    
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_id_{ts()}.wav")
    file.save(temp_path)
    
    try:
        vp = get_processor()
        models = load_user_voice_models(user['id'])
        vp.speaker_models = models
        
        result = vp.identify_speaker(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.route('/api/identify_record', methods=['POST'])
def identify_record():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio blob'}), 400
    raw = request.files['audio'].read()
    if len(raw) < 500:
        return jsonify({'error': 'Recording too short'}), 400
    try:
        wav_bytes = convert_to_wav(raw)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_id_rec_{ts()}.wav")
        with open(temp_path, 'wb') as f:
            f.write(wav_bytes)
        
        vp = get_processor()
        models = load_user_voice_models(user['id'])
        vp.speaker_models = models
        
        result = vp.identify_speaker(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ============================================================================
# ROUTES API - SPEAKERS
# ============================================================================

@app.route('/api/speakers', methods=['GET'])
def list_speakers():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    models = load_user_voice_models(user['id'])
    
    speakers = []
    for name, model in models.items():
        speakers.append({
            'name': name,
            'n_samples': model.get('n_samples', 0),
            'training_date': model.get('training_date', ''),
            'intra_similarity_mean': model.get('intra_similarity_mean', 0)
        })
    
    return jsonify({'speakers': speakers})

@app.route('/api/delete_speaker/<name>', methods=['DELETE'])
def delete_speaker(name):
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        file_path = f"{user['id']}/models/{name}.pkl"
        supabase.storage.from_("voice-models").remove([file_path])
        
        vp = get_processor()
        if name in vp.speaker_models:
            del vp.speaker_models[name]
        
        return jsonify({'message': f"'{name}' deleted"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ROUTES API - HEALTH
# ============================================================================

@app.route('/api/health')
def health():
    vp = get_processor()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(vp.speaker_models),
        'device': vp.device,
    })

# ============================================================================
# ROUTES API - COMPARE
# ============================================================================

@app.route('/api/compare', methods=['POST'])
def compare_voices():
    user = require_auth()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    
    return jsonify({'error': 'Compare à implémenter avec Supabase'}), 501

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print("=" * 55)
    print("  VoiceID — Neural Voice Recognition with Supabase")
    print("=" * 55)
    print(f"  Listening on port {port}")
    print("=" * 55)
    app.run(debug=False, host='0.0.0.0', port=port)
