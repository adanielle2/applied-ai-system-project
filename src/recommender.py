import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Score every song against the user profile and return the top k."""
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        scored: List[Tuple[Song, float]] = []
        for song in self.songs:
            song_dict = {
                "id": song.id,
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "acousticness": song.acousticness,
            }
            total_score, _ = score_song(prefs, song_dict)
            scored.append((song, total_score))

        scored.sort(key=lambda x: (-x[1], x[0].id))
        return [s for s, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why this song fits the user."""
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dict = {
            "id": song.id,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "acousticness": song.acousticness,
        }
        _, reasons = score_song(prefs, song_dict)
        return " | ".join(reasons) if reasons else "No matching signals found."

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with typed fields."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences; return (total_score, reasons)."""
    # ── EXPERIMENT: Weight Shift ───────────────────────────────────────────────
    # Original weights:  genre +3.0 · energy ×2.0  → max score 8.5
    # Experiment weights: genre +1.5 · energy ×4.0  → max score 9.0
    #
    # Math check:
    #   energy range is [0.0, 1.0], so abs(song.energy - target) ∈ [0.0, 1.0]
    #   → (1.0 - delta) ∈ [0.0, 1.0]  → ×4.0 keeps energy_points ∈ [0.0, 4.0] ✓
    #   All contributions remain non-negative; total score ∈ [0.0, 9.0] ✓
    # ──────────────────────────────────────────────────────────────────────────
    GENRE_WEIGHT  = 1.5   # was 3.0
    ENERGY_SCALE  = 4.0   # was 2.0
    MOOD_WEIGHT   = 2.0   # unchanged
    ACOUSTIC_HI   = 1.5   # unchanged
    ACOUSTIC_LO   = 1.0   # unchanged

    score = 0.0
    reasons = []

    # Genre match
    if song["genre"] == user_prefs.get("genre", ""):
        score += GENRE_WEIGHT
        reasons.append(f"genre match: {song['genre']} (+{GENRE_WEIGHT})")

    # Mood match
    if song["mood"] == user_prefs.get("mood", ""):
        score += MOOD_WEIGHT
        reasons.append(f"mood match: {song['mood']} (+{MOOD_WEIGHT})")

    # Energy proximity
    target_energy = user_prefs.get("energy", 0.5)
    energy_points = round((1.0 - abs(song["energy"] - target_energy)) * ENERGY_SCALE, 2)
    score += energy_points
    reasons.append(
        f"energy proximity: |{song['energy']} - {target_energy}| → +{energy_points}"
    )

    # Acoustic bonus
    likes_acoustic = user_prefs.get("likes_acoustic", None)
    if likes_acoustic is True and song["acousticness"] > 0.6:
        score += ACOUSTIC_HI
        reasons.append(f"acoustic match: acousticness {song['acousticness']} (+{ACOUSTIC_HI})")
    elif likes_acoustic is False and song["acousticness"] < 0.3:
        score += ACOUSTIC_LO
        reasons.append(f"non-acoustic match: acousticness {song['acousticness']} (+{ACOUSTIC_LO})")

    return round(score, 2), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score all songs, sort by score descending, and return the top k as (song, score, explanation)."""
    scored = []
    for song in songs:
        total_score, reasons = score_song(user_prefs, song)
        explanation = " | ".join(reasons)
        scored.append((song, total_score, explanation))

    scored.sort(key=lambda x: (-x[1], x[0]["id"]))
    return scored[:k]
