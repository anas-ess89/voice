"""
supabase_client.py - Client Supabase pour authentification et stockage cloud
"""

import os
from supabase import create_client, Client
from datetime import datetime
import numpy as np
import pickle
import io

SUPABASE_URL = "https://iwdpfafkcgnfkqcskpal.supabase.co"
SUPABASE_KEY = "sb_publishable_CQeLQ3eOqxM3XO7SQGrsAw_NXufiLFI"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("✅ Supabase client initialisé")


# ============================================================================
# AUTHENTIFICATION
# ============================================================================

def sign_up(email: str, password: str, username: str):
    try:
        print(f"📝 Tentative d'inscription: {email}")
        
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"username": username}}
        })
        
        user_id = auth_response.user.id
        
        existing = supabase.table("users").select("*").eq("id", user_id).execute()
        
        if not existing.data:
            supabase.table("users").insert({
                "id": user_id,
                "username": username,
                "email": email,
                "created_at": datetime.now().isoformat()
            }).execute()
            print(f"✅ Utilisateur créé: {user_id}")
        
        return {"success": True, "user_id": user_id}
    except Exception as e:
        print(f"❌ Erreur signup: {e}")
        return {"success": False, "message": str(e)}


def login(email: str, password: str):
    try:
        print(f"🔐 Tentative de connexion: {email}")
        
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        user = auth_response.user
        profile = supabase.table("users").select("*").eq("id", user.id).execute()
        
        username = profile.data[0]["username"] if profile.data else user.email
        
        return {
            "success": True,
            "user_id": user.id,
            "email": user.email,
            "username": username,
            "access_token": auth_response.session.access_token
        }
    except Exception as e:
        print(f"❌ Erreur login: {e}")
        return {"success": False, "message": str(e)}


def logout():
    try:
        supabase.auth.sign_out()
    except:
        pass
    return {"success": True}


def get_current_user():
    try:
        user = supabase.auth.get_user()
        if user:
            profile = supabase.table("users").select("*").eq("id", user.user.id).execute()
            return {
                "id": user.user.id,
                "email": user.user.email,
                "username": profile.data[0]["username"] if profile.data else user.user.email
            }
        return None
    except Exception as e:
        print(f"⚠️ Erreur get_current_user: {e}")
        return None


# ============================================================================
# STOCKAGE CLOUD - AUDIOS
# ============================================================================

def save_audio_to_cloud(user_id: str, audio_bytes: bytes, filename: str, file_type: str = "train"):
    try:
        file_path = f"{user_id}/{file_type}/{filename}"
        
        supabase.storage.from_("audio-files").upload(
            file_path,
            audio_bytes,
            {"content-type": "audio/wav", "upsert": "true"}
        )
        
        supabase.table("audio_files").insert({
            "user_id": user_id,
            "file_name": filename,
            "file_path": file_path,
            "file_type": file_type,
            "file_size": len(audio_bytes),
            "created_at": datetime.now().isoformat()
        }).execute()
        
        print(f"✅ Audio cloud: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Erreur audio: {e}")
        return None


# ============================================================================
# STOCKAGE CLOUD - EMBEDDINGS
# ============================================================================

def save_embedding_to_cloud(user_id: str, filename: str, embedding):
    try:
        emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        supabase.table("embeddings").insert({
            "user_id": user_id,
            "audio_file_name": filename,
            "embedding_vector": emb_list,
            "created_at": datetime.now().isoformat()
        }).execute()
        
        print(f"✅ Embedding cloud: {filename}")
        return True
    except Exception as e:
        print(f"❌ Erreur embedding: {e}")
        return False


# ============================================================================
# STOCKAGE CLOUD - MODÈLES VOCAUX
# ============================================================================

def save_voice_model_to_cloud(user_id: str, speaker_name: str, model: dict):
    """Sauvegarde le modèle vocal dans Supabase Storage"""
    try:
        file_path = f"{user_id}/models/{speaker_name}.pkl"
        model_bytes = pickle.dumps(model)
        
        supabase.storage.from_("voice-models").upload(
            file_path,
            model_bytes,
            {"content-type": "application/octet-stream", "upsert": "true"}
        )
        
        supabase.table("users").update({
            "voice_model_path": file_path,
            "voice_trained": True,
            "last_training": datetime.now().isoformat()
        }).eq("id", user_id).execute()
        
        print(f"✅ Modèle cloud: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Erreur sauvegarde modèle: {e}")
        return None


def load_user_voice_models(user_id: str):
    """Charge tous les modèles vocaux d'un utilisateur depuis le cloud"""
    try:
        files = supabase.storage.from_("voice-models").list(f"{user_id}/models/")
        
        models = {}
        for file in files:
            if file['name'].endswith('.pkl'):
                speaker_name = file['name'].replace('.pkl', '')
                file_path = f"{user_id}/models/{speaker_name}.pkl"
                data = supabase.storage.from_("voice-models").download(file_path)
                model = pickle.loads(data)
                models[speaker_name] = model
                print(f"📦 Modèle chargé: {speaker_name}")
        
        return models
    except Exception as e:
        print(f"⚠️ Aucun modèle pour user {user_id}: {e}")
        return {}


def get_user_speakers(user_id: str):
    """Récupère la liste des locuteurs d'un utilisateur"""
    try:
        files = supabase.storage.from_("voice-models").list(f"{user_id}/models/")
        
        speakers = []
        for file in files:
            if file['name'].endswith('.pkl'):
                speaker_name = file['name'].replace('.pkl', '')
                speakers.append(speaker_name)
        
        return speakers
    except Exception as e:
        print(f"⚠️ Erreur: {e}")
        return []


# ============================================================================
# ALIAS POUR COMPATIBILITÉ
# ============================================================================

save_audio_to_supabase = save_audio_to_cloud
save_embedding_to_supabase = save_embedding_to_cloud
save_voice_model_to_supabase = save_voice_model_to_cloud
get_embeddings_from_cloud = lambda user_id, filenames=None: []

# Export explicite des fonctions
__all__ = [
    'supabase',
    'sign_up', 'login', 'logout', 'get_current_user',
    'save_audio_to_cloud', 'save_embedding_to_cloud', 'save_voice_model_to_cloud',
    'load_user_voice_models', 'get_user_speakers',
    'save_audio_to_supabase', 'save_embedding_to_supabase', 'save_voice_model_to_supabase',
    'get_embeddings_from_cloud'
]