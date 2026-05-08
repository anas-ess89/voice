"""
train_model.py - Speaker model trainer using Resemblyzer LSTM embeddings
"""
import os
import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from voice_processor import VoiceProcessor
from hparams import verification_threshold, high_confidence_threshold


class VoiceTrainer:
    def __init__(self):
        self.voice_processor = VoiceProcessor()
        self.models_dir = Path('trained_models')
        self.models_dir.mkdir(exist_ok=True)
        self.uploads_dir = Path('uploads')

    # ─── Data Collection ────────────────────────────────────────────────────────

    def collect_speaker_audio(self, speaker_name: str, min_samples: int = 1) -> list:
        """Find audio files associated with a speaker name"""
        files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            files += list(self.uploads_dir.glob(f"*{speaker_name}*{ext}"))
        speaker_dir = self.uploads_dir / speaker_name
        if speaker_dir.exists():
            for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                files += list(speaker_dir.glob(f"*{ext}"))
        if len(files) < min_samples:
            raise ValueError(
                f"Need at least {min_samples} audio files for '{speaker_name}'. Found {len(files)}."
            )
        return files

    # ─── Training ───────────────────────────────────────────────────────────────

    def train_speaker(self, speaker_name: str, audio_files: list = None) -> dict:
        """Train a speaker model from a list of audio file paths"""
        if audio_files is None:
            audio_files = self.collect_speaker_audio(speaker_name)

        embeddings, successful = [], []
        print(f"[Trainer] Training '{speaker_name}' with {len(audio_files)} files...")

        for path in audio_files:
            emb = self.voice_processor.get_embedding(str(path))
            name = path.name if hasattr(path, 'name') else str(path)
            if emb is not None:
                embeddings.append(emb)
                successful.append(str(path))
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}")

        if len(embeddings) < 1:
            raise ValueError(f"Need at least 1 valid embedding. Got {len(embeddings)}.")

        arr = np.array(embeddings)
        mean_emb = arr.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb)

        intra = [float(np.dot(arr[i], arr[j]))
                 for i in range(len(arr)) for j in range(i + 1, len(arr))]

        model = {
            'speaker_name': speaker_name,
            'embeddings': arr.tolist(),
            'mean_embedding': mean_emb.tolist(),
            'n_samples': len(embeddings),
            'training_date': datetime.now().isoformat(),
            'intra_similarity_mean': float(np.mean(intra)) if intra else 0.0,
            'intra_similarity_std': float(np.std(intra)) if intra else 0.0,
            'files': successful,
        }

        model_path = self.models_dir / f"{speaker_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.voice_processor.speaker_models[speaker_name] = model

        return {
            'status': 'success',
            'speaker_name': speaker_name,
            'n_samples': len(embeddings),
            'intra_similarity_mean': model['intra_similarity_mean'],
            'model_path': str(model_path),
            'message': f"Model trained for '{speaker_name}' with {len(embeddings)} samples",
        }

    def train_multiple_speakers(self, speakers_data: dict) -> dict:
        results = {}
        for name, files in speakers_data.items():
            try:
                results[name] = self.train_speaker(name, files)
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
        return results

    # ─── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate_speaker(self, speaker_name: str, test_audio_paths: list) -> dict:
        """Quick evaluation: test files vs stored speaker model"""
        if speaker_name not in self.voice_processor.speaker_models:
            return {'error': f"Speaker '{speaker_name}' not found"}

        model = self.voice_processor.speaker_models[speaker_name]
        refs = [np.array(e) for e in model['embeddings']]

        scores = []
        for p in test_audio_paths:
            emb = self.voice_processor.get_embedding(str(p))
            if emb is not None:
                sim = float(np.mean([np.dot(emb, r) for r in refs]))
                scores.append({'file': str(p), 'similarity': sim,
                               'verified': sim >= verification_threshold})

        return {
            'speaker': speaker_name,
            'test_results': scores,
            'mean_similarity': float(np.mean([s['similarity'] for s in scores])) if scores else 0,
            'verified_count': sum(1 for s in scores if s['verified']),
        }

    # ─── Management ─────────────────────────────────────────────────────────────

    def get_trained_speakers(self) -> list:
        speakers = []
        for p in self.models_dir.glob('*.pkl'):
            try:
                with open(p, 'rb') as f:
                    m = pickle.load(f)
                speakers.append({
                    'name': m['speaker_name'],
                    'n_samples': m['n_samples'],
                    'training_date': m['training_date'],
                    'intra_similarity_mean': m.get('intra_similarity_mean', 0),
                })
            except Exception as e:
                print(f"[Trainer] Could not read {p}: {e}")
        return sorted(speakers, key=lambda x: x['training_date'], reverse=True)

    def delete_speaker(self, speaker_name: str) -> bool:
        path = self.models_dir / f"{speaker_name}.pkl"
        if path.exists():
            path.unlink()
        self.voice_processor.speaker_models.pop(speaker_name, None)
        return True

    def export_training_report(self) -> dict:
        speakers = self.get_trained_speakers()
        report = {
            'total_speakers': len(speakers),
            'speakers': speakers,
            'total_samples': sum(s['n_samples'] for s in speakers),
            'thresholds': {
                'verification': verification_threshold,
                'high_confidence': high_confidence_threshold,
            },
            'export_date': datetime.now().isoformat(),
        }
        try:
            report_path = self.models_dir / 'training_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            print(f"[Trainer] Could not export report: {e}")
        return report