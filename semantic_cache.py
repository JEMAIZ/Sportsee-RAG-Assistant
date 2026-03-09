# semantic_cache.py
"""
Cache sémantique Redis pour SportSee RAG Assistant.

Fonctionnement :
  1. À chaque question, on calcule son embedding (MistralAI)
  2. On compare avec les embeddings en cache (similarité cosine)
  3. Si similarité > seuil → retourne la réponse en cache (pas d'appel LLM)
  4. Sinon → appel LLM normal, réponse stockée en cache

Clés Redis :
  cache:embeddings  → hash  {question_hash: embedding JSON}
  cache:responses   → hash  {question_hash: réponse JSON}
  cache:questions   → hash  {question_hash: question originale}
"""

import json
import hashlib
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Seuil de similarité cosine : 0.92 = questions très proches
SIMILARITY_THRESHOLD = 0.92
# TTL du cache : 24h (en secondes)
CACHE_TTL = 86400


def _cosine_similarity(a: list, b: list) -> float:
    """Similarité cosine entre deux vecteurs."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


def _hash_question(question: str) -> str:
    """Hash MD5 court d'une question (clé Redis)."""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


class SemanticCache:
    """
    Cache sémantique Redis.
    Utilise les embeddings Mistral pour trouver des questions similaires.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._redis = None
        self._embedder = None
        self._host = host
        self._port = port
        self._db = db
        self._available = False
        self._connect()

    def _connect(self):
        """Connexion Redis — non bloquante si Redis est absent."""
        try:
            import redis
            client = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            client.ping()
            self._redis = client
            self._available = True
            logger.info(f"SemanticCache connecté à Redis {self._host}:{self._port}")
        except Exception as e:
            logger.warning(f"Redis non disponible — cache désactivé : {e}")
            self._available = False

    def _get_embedder(self):
        """Charge l'embedder Mistral (lazy init)."""
        if self._embedder is None:
            try:
                from langchain_mistralai import MistralAIEmbeddings
                from utils.config import MISTRAL_API_KEY
                self._embedder = MistralAIEmbeddings(
                    api_key=MISTRAL_API_KEY,
                    model="mistral-embed",
                )
            except Exception as e:
                logger.error(f"Impossible de charger l'embedder : {e}")
                return None
        return self._embedder

    def _embed(self, text: str) -> Optional[list]:
        """Calcule l'embedding d'un texte."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            return embedder.embed_query(text)
        except Exception as e:
            logger.warning(f"Erreur embedding : {e}")
            return None

    def get(self, question: str) -> Optional[str]:
        """
        Cherche une réponse en cache pour une question similaire.
        Retourne la réponse si trouvée, None sinon.
        """
        if not self._available:
            return None

        query_embedding = self._embed(question)
        if query_embedding is None:
            return None

        try:
            # Récupère tous les embeddings stockés
            all_embeddings = self._redis.hgetall("cache:embeddings")
            if not all_embeddings:
                return None

            best_score = 0.0
            best_key   = None

            for key, emb_json in all_embeddings.items():
                stored_emb = json.loads(emb_json)
                score = _cosine_similarity(query_embedding, stored_emb)
                if score > best_score:
                    best_score = score
                    best_key   = key

            if best_score >= SIMILARITY_THRESHOLD and best_key:
                response_json = self._redis.hget("cache:responses", best_key)
                original_q    = self._redis.hget("cache:questions", best_key)
                if response_json:
                    response = json.loads(response_json)
                    logger.info(
                        f"Cache HIT (score={best_score:.3f}) "
                        f"— question similaire : '{original_q}'"
                    )
                    return response
        except Exception as e:
            logger.warning(f"Erreur lecture cache : {e}")

        return None

    def set(self, question: str, response: str) -> bool:
        """Stocke une question/réponse dans le cache."""
        if not self._available:
            return False

        embedding = self._embed(question)
        if embedding is None:
            return False

        try:
            key = _hash_question(question)
            self._redis.hset("cache:embeddings", key, json.dumps(embedding))
            self._redis.hset("cache:responses",  key, json.dumps(response))
            self._redis.hset("cache:questions",  key, question)
            # TTL sur les 3 hashes
            for hkey in ["cache:embeddings", "cache:responses", "cache:questions"]:
                self._redis.expire(hkey, CACHE_TTL)
            logger.info(f"Cache SET — question stockée : '{question[:60]}...'")
            return True
        except Exception as e:
            logger.warning(f"Erreur écriture cache : {e}")
            return False

    def clear(self) -> bool:
        """Vide le cache Redis."""
        if not self._available:
            return False
        try:
            self._redis.delete("cache:embeddings", "cache:responses", "cache:questions")
            logger.info("Cache Redis vidé.")
            return True
        except Exception as e:
            logger.warning(f"Erreur clear cache : {e}")
            return False

    def stats(self) -> dict:
        """Retourne les statistiques du cache."""
        if not self._available:
            return {"available": False}
        try:
            count = len(self._redis.hkeys("cache:embeddings"))
            return {
                "available": True,
                "entries":   count,
                "host":      f"{self._host}:{self._port}",
                "threshold": SIMILARITY_THRESHOLD,
                "ttl_hours": CACHE_TTL // 3600,
            }
        except Exception:
            return {"available": False}


# Instance globale — partagée dans tout le projet
_cache_instance: Optional[SemanticCache] = None

def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        import os
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        _cache_instance = SemanticCache(host=host, port=port)
    return _cache_instance
