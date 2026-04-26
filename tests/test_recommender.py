"""
Unit tests for the Music Recommender scoring logic and OOP interface.

Tests cover:
- score_song() with known inputs and mathematically expected outputs
- Recommender.recommend() returns correct sorted order
- Recommender.explain_recommendation() surfaces meaningful reasons
- Edge cases: unknown genre, neutral preferences, acoustic/non-acoustic bonuses
"""

import pytest
from src.recommender import Song, UserProfile, Recommender, score_song


# ── Fixtures ────────────────────────────────────────────────────────────────────

def make_catalog() -> list[Song]:
    return [
        Song(id=1, title="Pop Hit", artist="A", genre="pop", mood="happy",
             energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=2, title="Chill Loop", artist="B", genre="lofi", mood="chill",
             energy=0.4, tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9),
        Song(id=3, title="Rock Out", artist="C", genre="rock", mood="intense",
             energy=0.9, tempo_bpm=150, valence=0.4, danceability=0.6, acousticness=0.1),
        Song(id=4, title="Soft Folk", artist="D", genre="folk", mood="nostalgic",
             energy=0.3, tempo_bpm=70, valence=0.65, danceability=0.45, acousticness=0.95),
    ]


# ── score_song() tests ──────────────────────────────────────────────────────────

class TestScoreSong:
    def test_perfect_match_scores_maximum(self):
        """A song matching genre + mood + energy + acoustic preference should score 9.0."""
        prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True}
        song = {"id": 2, "genre": "lofi", "mood": "chill", "energy": 0.4, "acousticness": 0.9}
        score, reasons = score_song(prefs, song)
        # genre(1.5) + mood(2.0) + energy(4.0) + acoustic(1.5) = 9.0
        assert score == pytest.approx(9.0, abs=0.01)
        assert len(reasons) == 4

    def test_no_match_scores_near_zero_energy_only(self):
        """A song with no genre/mood match and worst energy still scores energy contribution."""
        prefs = {"genre": "pop", "mood": "happy", "energy": 1.0, "likes_acoustic": False}
        # energy 0.0 → |0.0 - 1.0| = 1.0 → (1 - 1) * 4 = 0.0
        song = {"id": 99, "genre": "blues", "mood": "sad", "energy": 0.0, "acousticness": 0.5}
        score, reasons = score_song(prefs, song)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_genre_match_adds_correct_weight(self):
        prefs = {"genre": "rock", "mood": "happy", "energy": 0.5, "likes_acoustic": None}
        song = {"id": 3, "genre": "rock", "mood": "sad", "energy": 0.5, "acousticness": 0.5}
        score, reasons = score_song(prefs, song)
        # genre(1.5) + mood(0) + energy(4.0) = 5.5
        assert score == pytest.approx(5.5, abs=0.01)
        assert any("genre match" in r for r in reasons)

    def test_mood_match_adds_correct_weight(self):
        prefs = {"genre": "jazz", "mood": "chill", "energy": 0.5, "likes_acoustic": None}
        song = {"id": 5, "genre": "pop", "mood": "chill", "energy": 0.5, "acousticness": 0.5}
        score, reasons = score_song(prefs, song)
        # mood(2.0) + energy(4.0) = 6.0
        assert score == pytest.approx(6.0, abs=0.01)
        assert any("mood match" in r for r in reasons)

    def test_energy_proximity_is_continuous(self):
        """Energy score should decrease continuously as the gap widens."""
        prefs = {"genre": "", "mood": "", "energy": 0.5, "likes_acoustic": None}
        close = {"id": 1, "genre": "x", "mood": "x", "energy": 0.6, "acousticness": 0.5}
        far = {"id": 2, "genre": "x", "mood": "x", "energy": 0.9, "acousticness": 0.5}
        close_score, _ = score_song(prefs, close)
        far_score, _ = score_song(prefs, far)
        assert close_score > far_score

    def test_acoustic_bonus_high_acousticness(self):
        """User who likes acoustic should get +1.5 when song acousticness > 0.6."""
        prefs = {"genre": "", "mood": "", "energy": 0.5, "likes_acoustic": True}
        song = {"id": 1, "genre": "folk", "mood": "chill", "energy": 0.5, "acousticness": 0.8}
        score, reasons = score_song(prefs, song)
        assert any("acoustic match" in r for r in reasons)
        # energy(4.0) + acoustic(1.5) = 5.5
        assert score == pytest.approx(5.5, abs=0.01)

    def test_non_acoustic_bonus(self):
        """User who dislikes acoustic should get +1.0 when song acousticness < 0.3."""
        prefs = {"genre": "", "mood": "", "energy": 0.5, "likes_acoustic": False}
        song = {"id": 1, "genre": "edm", "mood": "euphoric", "energy": 0.5, "acousticness": 0.05}
        score, reasons = score_song(prefs, song)
        assert any("non-acoustic match" in r for r in reasons)
        # energy(4.0) + non-acoustic(1.0) = 5.0
        assert score == pytest.approx(5.0, abs=0.01)

    def test_unknown_genre_earns_no_genre_bonus(self):
        """Requesting a genre not in the catalog means no song ever gets the genre bonus."""
        prefs = {"genre": "k-pop", "mood": "", "energy": 0.5, "likes_acoustic": None}
        song = {"id": 1, "genre": "pop", "mood": "", "energy": 0.5, "acousticness": 0.5}
        score, reasons = score_song(prefs, song)
        assert not any("genre match" in r for r in reasons)


# ── Recommender.recommend() tests ───────────────────────────────────────────────

class TestRecommender:
    def test_returns_correct_count(self):
        rec = Recommender(make_catalog())
        user = UserProfile("pop", "happy", 0.8, False)
        results = rec.recommend(user, k=2)
        assert len(results) == 2

    def test_best_match_is_first(self):
        """Pop/happy/high-energy user should get the pop song first."""
        rec = Recommender(make_catalog())
        user = UserProfile("pop", "happy", 0.8, False)
        results = rec.recommend(user, k=4)
        assert results[0].genre == "pop"
        assert results[0].mood == "happy"

    def test_lofi_user_gets_lofi_first(self):
        rec = Recommender(make_catalog())
        user = UserProfile("lofi", "chill", 0.4, True)
        results = rec.recommend(user, k=4)
        assert results[0].genre == "lofi"

    def test_results_are_sorted_descending_by_score(self):
        """Verify that every consecutive pair is in non-increasing score order."""
        rec = Recommender(make_catalog())
        user = UserProfile("folk", "nostalgic", 0.3, True)
        results = rec.recommend(user, k=4)
        prefs = {"genre": user.favorite_genre, "mood": user.favorite_mood,
                 "energy": user.target_energy, "likes_acoustic": user.likes_acoustic}
        scores = []
        for s in results:
            sd = {"id": s.id, "genre": s.genre, "mood": s.mood,
                  "energy": s.energy, "acousticness": s.acousticness}
            score, _ = score_song(prefs, sd)
            scores.append(score)
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_catalog_returns_all_songs(self):
        rec = Recommender(make_catalog())
        user = UserProfile("pop", "happy", 0.8, False)
        results = rec.recommend(user, k=100)
        assert len(results) == len(make_catalog())

    def test_explain_returns_non_empty_string(self):
        rec = Recommender(make_catalog())
        user = UserProfile("pop", "happy", 0.8, False)
        song = rec.songs[0]
        explanation = rec.explain_recommendation(user, song)
        assert isinstance(explanation, str)
        assert explanation.strip() != ""

    def test_explain_mentions_matching_signals(self):
        """Explanation for a genre+mood match should mention both signals."""
        rec = Recommender(make_catalog())
        user = UserProfile("pop", "happy", 0.8, False)
        pop_song = next(s for s in rec.songs if s.genre == "pop")
        explanation = rec.explain_recommendation(user, pop_song)
        assert "genre match" in explanation
        assert "mood match" in explanation
