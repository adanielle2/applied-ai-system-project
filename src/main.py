"""
Command line runner for the Music Recommender Simulation.

Runs six user profiles — three standard and three adversarial — and
prints the top 5 recommendations with scores and reasons for each.
"""

from .recommender import load_songs, recommend_songs

# ── Standard profiles ──────────────────────────────────────────────────────────

PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": False,
    },

    # ── Adversarial / edge-case profiles ───────────────────────────────────────

    # Conflicting preferences: high energy but sad mood.
    # No song in the catalog pairs high energy with sadness, so the scorer
    # must choose between rewarding energy (continuous) or mood (categorical).
    # Expected weakness: mood never matches, so genre + energy dominate.
    "Conflicting: High-Energy + Sad": {
        "genre": "blues",
        "mood": "sad",
        "energy": 0.9,
        "likes_acoustic": False,
    },

    # Impossible genre: asks for a genre that does not exist in the catalog.
    # No song ever earns the +3.0 genre bonus, so the ranking falls back
    # entirely on mood, energy, and acoustic signals.
    # Expected weakness: exposes how much genre dominates scoring.
    "Unknown Genre": {
        "genre": "k-pop",
        "mood": "happy",
        "energy": 0.75,
        "likes_acoustic": False,
    },

    # Perfectly middle-of-the-road preferences (energy = 0.5, no genre/mood bias).
    # Every song gets a similar energy score, so tiny differences in acousticness
    # or a single genre/mood match decide the entire ranking.
    # Expected weakness: results feel arbitrary; the system has no strong signal.
    "Perfectly Neutral": {
        "genre": "",
        "mood": "",
        "energy": 0.5,
        "likes_acoustic": None,
    },
}


def run_profile(name: str, prefs: dict, songs: list) -> None:
    """Print top-5 recommendations and scores for one user profile."""
    print(f"\n{'═' * 60}")
    print(f"  Profile: {name}")
    print(f"  Prefs:   {prefs}")
    print(f"{'═' * 60}")
    results = recommend_songs(prefs, songs, k=5)
    if not results:
        print("  No recommendations returned.")
        return
    for rank, (song, score, explanation) in enumerate(results, start=1):
        print(f"  {rank}. {song['title']} ({song['genre']} · {song['mood']}) — {score:.2f}")
        print(f"     {explanation}")


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs.\n")
    for name, prefs in PROFILES.items():
        run_profile(name, prefs, songs)
    print()


if __name__ == "__main__":
    main()
