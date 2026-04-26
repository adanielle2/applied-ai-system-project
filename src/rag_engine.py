"""
rag_engine.py — Retrieval-Augmented Generation engine for the Music Recommender.

This module integrates RAG into the recommendation pipeline:
1. Loads song metadata + rich text descriptions into a ChromaDB vector store
2. Embeds user queries using sentence-transformers
3. Retrieves semantically relevant songs from the vector store
4. Passes retrieved context to an LLM (OpenAI-compatible) for intelligent ranking + explanations
5. Falls back to the original scoring algorithm if the LLM is unavailable

The RAG feature is fully integrated — the LLM's recommendations are grounded
in the retrieved song data, not generated from general knowledge.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .recommender import load_songs, score_song

# ── Logging setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("rag_engine")
logger.setLevel(logging.DEBUG)

# File handler for persistent logs
_fh = logging.FileHandler("rag_engine.log", mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
logger.addHandler(_fh)

# Console handler for visibility
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
logger.addHandler(_ch)


# ── Constants ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "song_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # lightweight, fast, good quality


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for music recommendations.

    Combines semantic search (ChromaDB + sentence-transformers) with LLM reasoning
    (OpenAI-compatible API) to produce context-grounded recommendations.
    """

    def __init__(
        self,
        songs_csv_path: str = "data/songs.csv",
        descriptions_path: str = "knowledge_base/song_descriptions.json",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
    ):
        logger.info("Initializing RAG engine...")

        # Load song catalog
        self.songs = load_songs(songs_csv_path)
        self.songs_by_id = {s["id"]: s for s in self.songs}
        logger.info(f"Loaded {len(self.songs)} songs from {songs_csv_path}")

        # Load rich descriptions
        self.descriptions = self._load_descriptions(descriptions_path)
        logger.info(f"Loaded {len(self.descriptions)} song descriptions")

        # Initialize embedding model directly (bypasses chromadb's import check
        # which fails when tensorflow's broken numpy dependency is present)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self._st_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embed_fn = lambda texts: self._st_model.encode(
            texts, convert_to_numpy=False, convert_to_tensor=True
        ).tolist()
        logger.info(f"Embedding model loaded")

        # Initialize ChromaDB (persistent local storage)
        self.chroma_client = chromadb.Client()
        self._build_vector_store()

        # Initialize LLM client (optional — graceful fallback if not configured)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_client = None
        self.model_name = model_name
        if api_key:
            try:
                self.llm_client = OpenAI(
                    api_key=api_key,
                    base_url=openai_base_url or os.getenv("OPENAI_BASE_URL"),
                )
                logger.info(f"LLM client initialized (model: {model_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
        else:
            logger.warning(
                "No OPENAI_API_KEY found. LLM features disabled — "
                "will fall back to score-based ranking with RAG retrieval."
            )

    # ── Data Loading ───────────────────────────────────────────────────────────

    def _load_descriptions(self, path: str) -> Dict[int, str]:
        """Load song descriptions from JSON file. Returns {song_id: description}."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {item["id"]: item["description"] for item in data}
        except FileNotFoundError:
            logger.warning(f"Descriptions file not found: {path}")
            return {}
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing descriptions file: {e}")
            return {}

    # ── Vector Store ───────────────────────────────────────────────────────────

    def _build_vector_store(self) -> None:
        """Build (or rebuild) the ChromaDB collection from song data + descriptions."""
        # Delete existing collection if it exists, then recreate
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        documents = []
        metadatas = []
        ids = []

        for song in self.songs:
            # Build a rich text document combining metadata + description
            desc = self.descriptions.get(song["id"], "")
            doc_text = (
                f"{song['title']} by {song['artist']}. "
                f"Genre: {song['genre']}. Mood: {song['mood']}. "
                f"Energy: {song['energy']}/1.0. Tempo: {song['tempo_bpm']} BPM. "
                f"Acousticness: {song['acousticness']}/1.0. "
                f"Valence: {song['valence']}/1.0. "
                f"Danceability: {song['danceability']}/1.0. "
                f"{desc}"
            )
            documents.append(doc_text)
            metadatas.append({
                "id": song["id"],
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "mood": song["mood"],
                "energy": song["energy"],
                "acousticness": song["acousticness"],
            })
            ids.append(str(song["id"]))

        embeddings = self.embed_fn(documents)
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        logger.info(f"Vector store built: {len(documents)} songs indexed")

    # ── Query Building ─────────────────────────────────────────────────────────

    def _build_query(self, user_prefs: Dict, free_text: str = "") -> str:
        """
        Convert user preferences + optional free-text into a search query string.
        This query will be embedded and used for semantic similarity search.
        """
        parts = []

        if free_text:
            parts.append(free_text)

        genre = user_prefs.get("genre", "")
        if genre:
            parts.append(f"{genre} music")

        mood = user_prefs.get("mood", "")
        if mood:
            parts.append(f"{mood} mood")

        energy = user_prefs.get("energy", 0.5)
        if energy > 0.7:
            parts.append("high energy intense upbeat")
        elif energy < 0.3:
            parts.append("calm low energy peaceful quiet")
        else:
            parts.append("moderate energy balanced")

        likes_acoustic = user_prefs.get("likes_acoustic")
        if likes_acoustic is True:
            parts.append("acoustic organic natural instruments")
        elif likes_acoustic is False:
            parts.append("electronic produced synthetic")

        query = ". ".join(parts) if parts else "recommend me some music"
        logger.debug(f"Built query: {query}")
        return query

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self, user_prefs: Dict, free_text: str = "", n_results: int = 10
    ) -> List[Dict]:
        """
        Retrieve the top n_results songs from the vector store based on
        semantic similarity to the user's preferences and free-text query.

        Returns a list of dicts: [{song_data, distance, document}, ...]
        """
        query = self._build_query(user_prefs, free_text)
        logger.info(f"Retrieving top {n_results} songs for query: {query[:80]}...")

        try:
            query_embedding = self.embed_fn([query])
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, len(self.songs)),
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for i, song_id_str in enumerate(results["ids"][0]):
                song_id = int(song_id_str)
                song_data = self.songs_by_id.get(song_id)
                if song_data:
                    retrieved.append({
                        "song": song_data,
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "document": results["documents"][0][i] if results["documents"] else "",
                    })

        logger.info(f"Retrieved {len(retrieved)} songs")
        for r in retrieved:
            logger.debug(
                f"  - {r['song']['title']} (distance: {r['distance']:.4f})"
            )

        return retrieved

    # ── LLM Ranking ────────────────────────────────────────────────────────────

    def _build_llm_prompt(
        self, user_prefs: Dict, free_text: str, retrieved_songs: List[Dict], k: int
    ) -> str:
        """Build the prompt that asks the LLM to rank and explain retrieved songs."""
        # Format user preferences
        pref_lines = []
        if user_prefs.get("genre"):
            pref_lines.append(f"- Favorite genre: {user_prefs['genre']}")
        if user_prefs.get("mood"):
            pref_lines.append(f"- Preferred mood: {user_prefs['mood']}")
        pref_lines.append(f"- Target energy level: {user_prefs.get('energy', 0.5)}/1.0")
        if user_prefs.get("likes_acoustic") is True:
            pref_lines.append("- Prefers acoustic/organic sounds")
        elif user_prefs.get("likes_acoustic") is False:
            pref_lines.append("- Prefers electronic/produced sounds")
        pref_str = "\n".join(pref_lines)

        # Format retrieved songs
        song_lines = []
        for i, r in enumerate(retrieved_songs, 1):
            s = r["song"]
            desc = self.descriptions.get(s["id"], "No description available.")
            song_lines.append(
                f"Song {i}: \"{s['title']}\" by {s['artist']}\n"
                f"  Genre: {s['genre']} | Mood: {s['mood']} | Energy: {s['energy']} | "
                f"Acousticness: {s['acousticness']} | Tempo: {s['tempo_bpm']} BPM\n"
                f"  Description: {desc}"
            )
        songs_str = "\n\n".join(song_lines)

        free_text_section = ""
        if free_text:
            free_text_section = f"\nThe user also said: \"{free_text}\"\n"

        prompt = f"""You are a music recommendation expert. A user has provided their preferences, and I have retrieved candidate songs from our catalog. Your job is to rank the top {k} songs that best match this user and explain WHY each song is a good fit.

USER PREFERENCES:
{pref_str}
{free_text_section}
CANDIDATE SONGS (retrieved from our catalog):
{songs_str}

INSTRUCTIONS:
1. Analyze how well each candidate matches the user's preferences
2. Consider genre, mood, energy level, acoustic preference, and any free-text context
3. Return EXACTLY {k} recommendations ranked from best to worst match
4. For each recommendation, provide a personalized 1-2 sentence explanation

Respond in this exact JSON format and nothing else:
{{
  "recommendations": [
    {{
      "song_id": <integer>,
      "title": "<song title>",
      "artist": "<artist name>",
      "rank": <1 to {k}>,
      "explanation": "<why this song fits the user>"
    }}
  ]
}}"""
        return prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM and return the response text. Returns None on failure."""
        if not self.llm_client:
            logger.warning("LLM client not available")
            return None

        try:
            logger.info(f"Calling LLM ({self.model_name})...")
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a music recommendation expert. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            result = response.choices[0].message.content
            logger.info("LLM response received")
            logger.debug(f"LLM raw response: {result[:200]}...")
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _parse_llm_response(self, response_text: str) -> Optional[List[Dict]]:
        """Parse the LLM's JSON response into a list of recommendation dicts."""
        try:
            # Strip markdown code fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]  # remove first line
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            recs = data.get("recommendations", [])
            logger.info(f"Parsed {len(recs)} recommendations from LLM response")
            return recs
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response was: {response_text[:300]}")
            return None

    # ── Fallback Ranking ───────────────────────────────────────────────────────

    def _fallback_ranking(
        self, user_prefs: Dict, retrieved_songs: List[Dict], k: int
    ) -> List[Dict]:
        """
        Fallback: use the original score_song algorithm to rank retrieved songs.
        This runs when the LLM is unavailable or fails.
        """
        logger.info("Using fallback scoring (original algorithm) on retrieved songs")
        scored = []
        for r in retrieved_songs:
            song = r["song"]
            total_score, reasons = score_song(user_prefs, song)
            desc = self.descriptions.get(song["id"], "")
            scored.append({
                "song_id": song["id"],
                "title": song["title"],
                "artist": song["artist"],
                "score": total_score,
                "explanation": f"Score: {total_score:.2f}. {' | '.join(reasons)}",
                "description": desc,
            })

        scored.sort(key=lambda x: (-x["score"], x["song_id"]))
        # Add rank
        for i, rec in enumerate(scored[:k], 1):
            rec["rank"] = i
        return scored[:k]

    # ── Main Recommend Method ──────────────────────────────────────────────────

    def recommend(
        self,
        user_prefs: Dict,
        free_text: str = "",
        k: int = 5,
        n_retrieve: int = 10,
    ) -> Dict:
        """
        Full RAG recommendation pipeline:
        1. Retrieve semantically relevant songs from the vector store
        2. Ask the LLM to rank and explain them (or fall back to scoring)
        3. Return structured results with full metadata

        Returns:
        {
            "query": str,
            "method": "rag_llm" | "rag_fallback",
            "recommendations": [
                {
                    "rank": int,
                    "song_id": int,
                    "title": str,
                    "artist": str,
                    "genre": str,
                    "mood": str,
                    "energy": float,
                    "explanation": str,
                }
            ],
            "retrieved_count": int,
        }
        """
        logger.info("=" * 60)
        logger.info(f"New recommendation request: k={k}, prefs={user_prefs}")
        if free_text:
            logger.info(f"Free text: {free_text}")

        # Step 1: Retrieve
        retrieved = self.retrieve(user_prefs, free_text, n_results=n_retrieve)
        if not retrieved:
            logger.error("No songs retrieved — returning empty results")
            return {
                "query": self._build_query(user_prefs, free_text),
                "method": "error",
                "recommendations": [],
                "retrieved_count": 0,
            }

        # Step 2: LLM ranking (with fallback)
        method = "rag_llm"
        recommendations = None

        if self.llm_client:
            prompt = self._build_llm_prompt(user_prefs, free_text, retrieved, k)
            llm_response = self._call_llm(prompt)
            if llm_response:
                parsed = self._parse_llm_response(llm_response)
                if parsed:
                    recommendations = parsed

        if recommendations is None:
            method = "rag_fallback"
            recommendations = self._fallback_ranking(user_prefs, retrieved, k)

        # Step 3: Enrich with full song metadata
        enriched = []
        for rec in recommendations:
            song_id = rec.get("song_id")
            song_data = self.songs_by_id.get(song_id, {})
            enriched.append({
                "rank": rec.get("rank", 0),
                "song_id": song_id,
                "title": rec.get("title", song_data.get("title", "Unknown")),
                "artist": rec.get("artist", song_data.get("artist", "Unknown")),
                "genre": song_data.get("genre", ""),
                "mood": song_data.get("mood", ""),
                "energy": song_data.get("energy", 0),
                "acousticness": song_data.get("acousticness", 0),
                "explanation": rec.get("explanation", ""),
            })

        result = {
            "query": self._build_query(user_prefs, free_text),
            "method": method,
            "recommendations": enriched,
            "retrieved_count": len(retrieved),
        }

        logger.info(f"Returning {len(enriched)} recommendations via {method}")
        return result
