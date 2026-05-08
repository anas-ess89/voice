"""
voice_processor.py - Voice processing with Resemblyzer LSTM
Uses pretrained.pt weights: LSTM(256) x 3 layers -> 256-dim L2-normalized embeddings
All audio params from hparams.py
"""
import sys
import os
import numpy as np
import torch
import librosa
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hparams import (
    sampling_rate, mel_n_channels, model_hidden_size,
    model_embedding_size, model_num_layers,
    verification_threshold, high_confidence_threshold,
    partials_n_frames, mel_window_length, mel_window_step,
    vad_window_length, vad_moving_average_width, vad_max_silence_length,
    audio_norm_target_dBFS
)


class VoiceProcessor:
    def __init__(self, model_path=None):
        self.encoder = None
        self.preprocess_wav = None
        self.speaker_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VoiceProcessor] Initializing on {self.device}...")
        self.load_encoder(model_path)
        self.load_speaker_models()
        print("[VoiceProcessor] Ready!")

    # ─── Model Loading ──────────────────────────────────────────────────────────

    def load_encoder(self, model_path=None):
        """Load Resemblyzer VoiceEncoder (LSTM x3, 256-dim embeddings)"""
        from resemblyzer import VoiceEncoder, preprocess_wav

        if model_path and Path(model_path).exists():
            self.encoder = VoiceEncoder(weights_fpath=model_path, device=self.device)
            print(f"[VoiceProcessor] Custom model loaded: {model_path}")
        else:
            self.encoder = VoiceEncoder(device=self.device)
            print("[VoiceProcessor] Default pretrained.pt loaded")

        self.preprocess_wav = preprocess_wav

        # Quick sanity check
        test_wav = np.zeros(sampling_rate * 2, dtype=np.float32)
        emb = self.encoder.embed_utterance(test_wav)
        print(f"[VoiceProcessor] Embedding dim: {emb.shape[0]} | "
              f"Architecture: LSTM({model_hidden_size}) x {model_num_layers} -> {model_embedding_size}d")

    def load_speaker_models(self):
        """Load all saved speaker .pkl models"""
        models_dir = Path('trained_models')
        if not models_dir.exists():
            return
        for f in models_dir.glob('*.pkl'):
            try:
                with open(f, 'rb') as fp:
                    data = pickle.load(fp)
                self.speaker_models[f.stem] = data
                print(f"[VoiceProcessor] Speaker loaded: {f.stem} ({data['n_samples']} samples)")
            except Exception as e:
                print(f"[VoiceProcessor] Warning: could not load {f}: {e}")

    # ─── Core Embedding ─────────────────────────────────────────────────────────

    def get_embedding(self, audio_path: str) -> np.ndarray | None:
        """
        Preprocess audio then extract 256-dim L2-normalized embedding via LSTM.
        Returns None if audio is too short or no voice detected.
        """
        if self.encoder is None or self.preprocess_wav is None:
            print("[VoiceProcessor] ERROR: encoder not loaded")
            return None
        try:
            wav = self.preprocess_wav(audio_path)          # resample + normalize + VAD
            duration = len(wav) / sampling_rate
            if duration < 0.5:
                print(f"[VoiceProcessor] Audio too short: {duration:.2f}s")
                return None

            embedding = self.encoder.embed_utterance(wav)   # (256,) L2-normed
            if embedding is None:
                print("[VoiceProcessor] No voice detected")
                return None
            return embedding
        except Exception as e:
            print(f"[VoiceProcessor] Error extracting embedding: {e}")
            return None

    # ─── Audio Analysis ─────────────────────────────────────────────────────────

    def process_audio(self, audio_path: str) -> dict:
        """Full audio analysis: stats + embedding + auto-identify"""
        wav, sr = librosa.load(audio_path, sr=None)
        duration = len(wav) / sr
        rms = float(np.sqrt(np.mean(wav ** 2)))
        peak = float(np.max(np.abs(wav)))

        embedding = self.get_embedding(audio_path)

        result = {
            'duration': duration,
            'sample_rate': sr,
            'rms_energy': rms,
            'peak_amplitude': peak,
            'embedding_extracted': embedding is not None,
            'embedding_dim': int(embedding.shape[0]) if embedding is not None else 0,
        }

        if embedding is not None and self.speaker_models:
            result['identified_speaker'] = self.identify_from_embedding(embedding)

        return result

    # ─── Comparison & Identification ────────────────────────────────────────────

    def compare_voices(self, path1: str, path2: str) -> float:
        """Cosine similarity between two audio files (embeddings are L2-normed → dot product = cos)"""
        e1 = self.get_embedding(path1)
        e2 = self.get_embedding(path2)
        if e1 is None or e2 is None:
            raise ValueError("Could not extract embeddings from one or both files")
        return float(np.dot(e1, e2))

    def identify_from_embedding(self, embedding: np.ndarray, threshold: float = None) -> dict:
        """Find best matching speaker from all stored models"""
        threshold = threshold or verification_threshold
        if not self.speaker_models:
            return {'speaker': None, 'confidence': 'low', 'similarity': 0}

        best_speaker, best_sim = None, -1.0
        all_scores = {}

        for name, model in self.speaker_models.items():
            refs = [np.array(e) for e in model['embeddings']]
            sim = float(np.mean([np.dot(embedding, r) for r in refs]))
            all_scores[name] = round(sim, 4)
            if sim > best_sim:
                best_sim, best_speaker = sim, name

        if best_sim >= high_confidence_threshold:
            confidence = 'high'
        elif best_sim >= threshold:
            confidence = 'medium'
        else:
            confidence = 'low'
            best_speaker = None

        return {
            'speaker': best_speaker,
            'confidence': confidence,
            'similarity': best_sim,
            'all_scores': all_scores,
        }

    def identify_speaker(self, audio_path: str) -> dict:
        embedding = self.get_embedding(audio_path)
        if embedding is None:
            return {'error': 'Could not extract voice embedding', 'speaker': None}
        return self.identify_from_embedding(embedding)

    # ─── Training ───────────────────────────────────────────────────────────────

    def train_speaker(self, speaker_name: str, audio_files_info: list,
                      uploaded_embeddings: dict = None) -> dict:
        """
        Build a speaker model from pre-uploaded embeddings or raw audio files.
        Uses mean of L2-normalized embeddings; re-normalizes the mean.
        """
        uploaded_embeddings = uploaded_embeddings or {}
        embeddings, successful_files = [], []

        for fi in audio_files_info:
            fname = fi.get('filename')
            if fname and fname in uploaded_embeddings:
                emb = uploaded_embeddings[fname].get('embedding')
                if emb is not None:
                    embeddings.append(emb)
                    successful_files.append(fname)
                    continue
            fpath = fi.get('path', '')
            if fpath and os.path.exists(fpath):
                emb = self.get_embedding(fpath)
                if emb is not None:
                    embeddings.append(emb)
                    successful_files.append(fpath)

        if not embeddings:
            raise ValueError("No valid embeddings extracted. Ensure audio files contain clear speech.")

        arr = np.array(embeddings)
        mean_emb = arr.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb)

        # Intra-class similarity (consistency metric)
        intra = [float(np.dot(arr[i], arr[j]))
                 for i in range(len(arr)) for j in range(i + 1, len(arr))]

        model = {
            'speaker_name': speaker_name,
            'embeddings': [e.tolist() for e in embeddings],
            'mean_embedding': mean_emb.tolist(),
            'n_samples': len(embeddings),
            'training_date': datetime.now().isoformat(),
            'intra_similarity_mean': float(np.mean(intra)) if intra else 0.0,
            'intra_similarity_std': float(np.std(intra)) if intra else 0.0,
            'files': successful_files,
            # hparams snapshot for traceability
            'hparams': {
                'sampling_rate': sampling_rate,
                'model_hidden_size': model_hidden_size,
                'model_embedding_size': model_embedding_size,
                'model_num_layers': model_num_layers,
                'partials_n_frames': partials_n_frames,
            }
        }

        Path('trained_models').mkdir(exist_ok=True)
        model_path = Path('trained_models') / f"{speaker_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        self.speaker_models[speaker_name] = model
        print(f"[VoiceProcessor] Model saved for '{speaker_name}' ({len(embeddings)} samples, "
              f"intra-sim={model['intra_similarity_mean']:.3f})")

        return {
            'status': 'success',
            'speaker_name': speaker_name,
            'n_samples': len(embeddings),
            'intra_similarity_mean': model['intra_similarity_mean'],
            'model_path': str(model_path),
        }

    # ─── Speaker Management ─────────────────────────────────────────────────────

    def get_trained_speakers(self) -> list:
        return sorted([
            {
                'name': n,
                'n_samples': m['n_samples'],
                'training_date': m['training_date'],
                'intra_similarity_mean': m.get('intra_similarity_mean', 0),
            }
            for n, m in self.speaker_models.items()
        ], key=lambda x: x['training_date'], reverse=True)

    def delete_speaker(self, speaker_name: str) -> bool:
        path = Path('trained_models') / f"{speaker_name}.pkl"
        if path.exists():
            path.unlink()
        self.speaker_models.pop(speaker_name, None)
        return True

    def batch_similarity_matrix(self, audio_paths: list) -> dict:
        embeddings, valid = [], []
        for p in audio_paths:
            e = self.get_embedding(p)
            if e is not None:
                embeddings.append(e)
                valid.append(p)
        if len(embeddings) < 2:
            return {'error': 'Need at least 2 valid audio files'}
        arr = np.array(embeddings)
        return {
            'similarity_matrix': (arr @ arr.T).tolist(),
            'files': valid,
        }
    def load_models_from_cloud(self, user_id: str):
        """Charge les modèles vocaux de l'utilisateur depuis Supabase"""
        from supabase_client import load_user_voice_models
        
        try:
            models = load_user_voice_models(user_id)
            self.speaker_models = models
            for name, model in models.items():
                print(f"[VoiceProcessor] Speaker loaded from cloud: {name} ({model['n_samples']} samples)")
            return models
        except Exception as e:
            print(f"[VoiceProcessor] No models for user {user_id}: {e}")
            self.speaker_models = {}
            return {}
        




def load_models_from_cloud(self, user_id: str = None):
    """Charge les modèles depuis Supabase Storage"""
    from supabase_client import supabase
    
    try:
        # Lister tous les modèles
        files = supabase.storage.from_("voice-models").list("")
        
        for file in files:
            if file['name'].endswith('.pkl'):
                try:
                    data = supabase.storage.from_("voice-models").download(file['name'])
                    model = pickle.loads(data)
                    speaker_name = model.get('speaker_name', file['name'].replace('.pkl', ''))
                    self.speaker_models[speaker_name] = model
                    print(f"📦 Modèle cloud chargé: {speaker_name}")
                except Exception as e:
                    print(f"⚠️ Erreur chargement {file['name']}: {e}")
    except Exception as e:
        print(f"⚠️ Aucun modèle cloud trouvé: {e}")