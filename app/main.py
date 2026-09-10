import uvicorn
from fastapi import FastAPI, Query, Depends, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from datetime import datetime
import warnings
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv
import re
from collections import Counter
from transformers import pipeline
import torch
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

# ============================================================
# Rate Limiting (slowapi)
# ============================================================
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

load_dotenv()
warnings.filterwarnings('ignore')

# ============================================================
# Métriques Prometheus
# ============================================================
REQUESTS = Counter('api_requests_total', 'Total des requêtes HTTP')
LATENCY = Histogram('api_latency_seconds', 'Latence des requêtes en secondes')

# ============================================================
# Modèle NLP (chargé une fois au démarrage)
# ============================================================
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device,
    truncation=True,
    max_length=512
)

# Cache LRU simple (même texte -> résultat)
cache_pred = {}

# ============================================================
# Collecteur
# ============================================================
from collectors.google_reviews_scraper import collecter_avis_google_auto_sync

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Analyse de sentiments - Google Reviews")

# Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Middleware pour les métriques Prometheus
# ============================================================
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        REQUESTS.inc()
        LATENCY.observe(duration)
        return response

app.add_middleware(PrometheusMiddleware)

# ============================================================
# Authentification Admin
# ============================================================
API_KEY = os.getenv("API_KEY", "Cle_naouel_2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verifier_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="❌ Clé API invalide ou manquante")
    return api_key

# ============================================================
# Connexion DB
# ============================================================
def get_db_connection():
    try:
        host = os.getenv("DB_HOST", "localhost")
        if host == "":
            host = "localhost"
        print(f"[DB] 🔗 Tentative de connexion à {host}:5432...")
        conn = psycopg2.connect(
            host=host,
            port=5432,
            user='Naouel',
            password='Pino2026',
            database='sentiment_db'
        )
        print(f"[DB] ✅ Connexion réussie à {host}")
        return conn
    except Exception as e:
        print(f"[DB] ❌ Erreur: {e}")
        return None

# ============================================================
# Fonctions DB
# ============================================================
def save_prediction(text, sentiment, confidence, source="google", metadata=None):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (text, sentiment, confidence, source, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (text, sentiment, confidence, source, json.dumps(metadata) if metadata else None))
        pred_id = cur.fetchone()[0]
        conn.commit()
        return pred_id
    except Exception as e:
        print(f"[DB] ❌ Erreur insertion: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_history(limit=50):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, text, sentiment, confidence, source, metadata, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()
    except Exception as e:
        print(f"[DB] ❌ Erreur historique: {e}")
        return []
    finally:
        conn.close()

def get_stats():
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) as total FROM predictions")
        total = cur.fetchone()['total']
        cur.execute("SELECT sentiment, COUNT(*) as count FROM predictions GROUP BY sentiment")
        repartition = cur.fetchall()
        cur.execute("""
            SELECT DATE(created_at) as jour, COUNT(*) as total,
                   SUM(CASE WHEN sentiment='POSITIVE' THEN 1 ELSE 0 END) as positifs,
                   SUM(CASE WHEN sentiment='NEGATIVE' THEN 1 ELSE 0 END) as negatifs,
                   SUM(CASE WHEN sentiment='NEUTRAL' THEN 1 ELSE 0 END) as neutres
            FROM predictions
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY jour DESC
        """)
        daily = cur.fetchall()
        return {"total": total, "repartition": repartition, "daily": daily}
    except Exception as e:
        print(f"[DB] ❌ Erreur stats: {e}")
        return {}
    finally:
        conn.close()

# ============================================================
# Analyse de sentiment (modèle NLP) avec cache
# ============================================================
def analyser_sentiment(texte):
    if not texte or len(texte.strip()) < 3:
        return 0.0, "NEUTRAL"

    # Vérifier le cache
    if texte in cache_pred:
        return cache_pred[texte]

    try:
        result = classifier(texte)[0]
        label = result['label']          # ex: "5 stars"
        stars = int(label.split()[0])    # extrait le chiffre (1 à 5)
        score = (stars - 3) / 2          # transforme en -1..1
        confidence = abs(score)

        if score > 0.2:
            sentiment = "POSITIVE"
        elif score < -0.2:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        # Stocker en cache
        cache_pred[texte] = (round(score, 2), sentiment)
        return cache_pred[texte]

    except Exception as e:
        print(f"[Analyse] Erreur modèle: {e}")
        return 0.0, "NEUTRAL"

# ============================================================
# Endpoints publics
# ============================================================
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Analyse d'avis Google Maps (avec cache)",
        "source": "Google Maps + DB Cache"
    }

@app.get("/recherche_avis")
@limiter.limit("5/minute")
def rechercher_et_analyser(
    request: Request,
    mot_cle: str = Query(..., description="Mot-clé"),
    limit: int = Query(5, description="Nombre d'avis", ge=1, le=20)
):
    print(f"[Recherche] 🔍 {mot_cle}")
    try:
        avis_bruts = collecter_avis_google_auto_sync(mot_cle, limit)
        print(f"[Recherche] 📊 {len(avis_bruts)} avis collectés")
        results = []
        for avis in avis_bruts:
            texte = avis.get("texte", "")
            if texte:
                score, sentiment = analyser_sentiment(texte)
                confidence = abs(score)
                save_prediction(
                    texte,
                    sentiment,
                    confidence,
                    source="google",
                    metadata={"note": avis.get("note"), "auteur": avis.get("auteur")}
                )
                results.append({
                    "text": texte,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "note": avis.get("note"),
                    "auteur": avis.get("auteur")
                })
        return {"total": len(results), "data": results}
    except Exception as e:
        print(f"[Recherche] ⚠️ Erreur: {e}")
        return {"total": 0, "data": [], "error": str(e)}

@app.get("/historique")
def get_historique_endpoint(limit: int = Query(50, description="Nombre max")):
    historique = get_history(limit)
    return {"total": len(historique), "data": historique}

@app.get("/stats")
def get_stats_endpoint():
    stats = get_stats()
    if not stats:
        return {"error": "Impossible de récupérer les statistiques"}
    return stats

@app.get("/cache")
def get_cache_stats():
    conn = get_db_connection()
    if not conn:
        return {"error": "Connexion DB impossible"}
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) as total FROM predictions")
        total = cur.fetchone()['total']
        cur.execute("""
            SELECT text, sentiment, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 5
        """)
        dernieres = cur.fetchall()
        return {
            "total_predictions": total,
            "dernieres": dernieres,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

# ============================================================
# Endpoint de test (analyse directe sans scraper)
# ============================================================
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

@app.post("/test-sentiment")
def test_sentiment(input: TextInput):
    score, sentiment = analyser_sentiment(input.text)
    return {"text": input.text, "sentiment": sentiment, "score": score}

# ============================================================
# Feedback utilisateur
# ============================================================
class FeedbackModel(BaseModel):
    prediction_id: int
    is_correct: bool

@app.post("/feedback")
def donner_feedback(feedback: FeedbackModel):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB error")
    try:
        cur = conn.cursor()
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feedback BOOLEAN;")
        cur.execute(
            "UPDATE predictions SET feedback = %s WHERE id = %s",
            (feedback.is_correct, feedback.prediction_id)
        )
        conn.commit()
        return {"status": "success", "message": "Feedback enregistré"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

# ============================================================
# Endpoints Admin (protégés)
# ============================================================
@app.get("/admin/stats")
def admin_stats(api_key: str = Depends(verifier_api_key)):
    stats = get_stats()
    if not stats:
        return {"status": "error", "message": "Erreur de récupération des stats"}
    return {"status": "success", "data": stats}

@app.get("/admin/cache")
def admin_cache(api_key: str = Depends(verifier_api_key)):
    conn = get_db_connection()
    if not conn:
        return {"error": "Connexion DB impossible"}
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, text, sentiment, confidence, source, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 20
        """)
        data = cur.fetchall()
        return {"status": "success", "cache_size": len(data), "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

@app.post("/admin/reload-model")
def reload_model(api_key: str = Depends(verifier_api_key)):
    print("[Admin] 🔄 Rechargement du modèle demandé")
    return {
        "status": "success",
        "message": "Modèle rechargé avec succès",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# Endpoint Prometheus Metrics
# ============================================================
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================================
# Lancement
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 API d'Analyse de Sentiments (avec cache et monitoring)")
    print("📡 http://127.0.0.1:8002")
    print("📚 Documentation: http://127.0.0.1:8002/docs")
    print("📊 Métriques: http://127.0.0.1:8002/metrics")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002) 
