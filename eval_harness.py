"""
eval_harness.py — Automated evaluation script for the Music Recommender system.

Runs the recommender against a fixed set of predefined test cases and prints
a pass/fail report with confidence scores. Each test case specifies user
preferences and one or more expectations about the output (e.g., which genre
should appear first, whether a specific song should rank in the top results).

Usage:
    python3 eval_harness.py

No arguments required. Results are printed to stdout and also logged to
eval_results.log for persistent record-keeping.
"""

import csv
import logging
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.recommender import load_songs, score_song, recommend_songs

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("eval_results.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("eval_harness")


# ── Test Case Definition ─────────────────────────────────────────────────────

@dataclass
class TestCase:
    """A single evaluation scenario."""
    name: str
    user_prefs: dict
    k: int
    # Each check is a function that takes the ranked result list and returns
    # (passed: bool, reason: str).  A case passes only if ALL checks pass.
    checks: List[Callable]


def check_top_genre(expected_genre: str) -> Callable:
    """The #1 recommendation must belong to the expected genre."""
    def _check(results):
        top = results[0][0] if results else None
        if top and top["genre"] == expected_genre:
            return True, f"top song is genre '{expected_genre}' ✓"
        actual = top["genre"] if top else "none"
        return False, f"expected genre '{expected_genre}' first, got '{actual}'"
    return _check


def check_top_mood(expected_mood: str) -> Callable:
    """The #1 recommendation must have the expected mood."""
    def _check(results):
        top = results[0][0] if results else None
        if top and top["mood"] == expected_mood:
            return True, f"top song has mood '{expected_mood}' ✓"
        actual = top["mood"] if top else "none"
        return False, f"expected mood '{expected_mood}' first, got '{actual}'"
    return _check


def check_top_k_contains_genre(genre: str) -> Callable:
    """At least one of the top-k results must belong to the expected genre."""
    def _check(results):
        genres = [r[0]["genre"] for r in results]
        if genre in genres:
            return True, f"genre '{genre}' present in top-{len(results)} ✓"
        return False, f"genre '{genre}' missing from top-{len(results)}: {genres}"
    return _check


def check_top_score_above(threshold: float) -> Callable:
    """The top song's score must exceed the threshold."""
    def _check(results):
        if not results:
            return False, "no results returned"
        top_score = results[0][1]
        if top_score >= threshold:
            return True, f"top score {top_score:.2f} ≥ {threshold} ✓"
        return False, f"top score {top_score:.2f} < {threshold}"
    return _check


def check_result_count(expected_k: int) -> Callable:
    """The result list must have exactly expected_k items."""
    def _check(results):
        n = len(results)
        if n == expected_k:
            return True, f"returned {n} results ✓"
        return False, f"expected {expected_k} results, got {n}"
    return _check


def check_scores_descending() -> Callable:
    """Every consecutive score pair must be in non-increasing order."""
    def _check(results):
        scores = [r[1] for r in results]
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                return False, f"scores not descending at index {i}: {scores}"
        return True, "scores are in descending order ✓"
    return _check


# ── Test Suite ───────────────────────────────────────────────────────────────

TEST_CASES: List[TestCase] = [
    TestCase(
        name="High-energy pop user gets pop song first",
        user_prefs={"genre": "pop", "mood": "happy", "energy": 0.9, "likes_acoustic": False},
        k=5,
        checks=[
            check_result_count(5),
            check_top_genre("pop"),
            check_top_mood("happy"),
            check_top_score_above(6.0),
            check_scores_descending(),
        ],
    ),
    TestCase(
        name="Chill lofi user gets lofi song first",
        user_prefs={"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True},
        k=5,
        checks=[
            check_result_count(5),
            check_top_genre("lofi"),
            check_top_score_above(7.0),
            check_scores_descending(),
        ],
    ),
    TestCase(
        name="Intense rock user gets rock song first",
        user_prefs={"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False},
        k=5,
        checks=[
            check_result_count(5),
            check_top_genre("rock"),
            check_top_score_above(6.0),
            check_scores_descending(),
        ],
    ),
    TestCase(
        name="Unknown genre falls back to energy/mood signals only",
        user_prefs={"genre": "k-pop", "mood": "happy", "energy": 0.75, "likes_acoustic": False},
        k=5,
        checks=[
            check_result_count(5),
            check_scores_descending(),
            # No k-pop in catalog — top result should still have a reasonable score
            check_top_score_above(4.0),
        ],
    ),
    TestCase(
        name="Conflicting preferences (high energy + sad) — top score still reasonable",
        user_prefs={"genre": "blues", "mood": "sad", "energy": 0.9, "likes_acoustic": False},
        k=5,
        checks=[
            check_result_count(5),
            check_scores_descending(),
            check_top_score_above(3.0),
        ],
    ),
    TestCase(
        name="Neutral user (no genre/mood, energy 0.5) returns 5 results in score order",
        user_prefs={"genre": "", "mood": "", "energy": 0.5, "likes_acoustic": None},
        k=5,
        checks=[
            check_result_count(5),
            check_scores_descending(),
        ],
    ),
    TestCase(
        name="Acoustic lover gets high-acousticness song in top 3",
        user_prefs={"genre": "", "mood": "", "energy": 0.3, "likes_acoustic": True},
        k=3,
        checks=[
            check_result_count(3),
            check_scores_descending(),
            # High-acousticness genres (folk, ambient, blues, lofi) should dominate;
            # folk edges out lofi here because it has closer energy proximity (0.31 vs 0.35–0.42)
            check_top_k_contains_genre("folk"),
        ],
    ),
    TestCase(
        name="EDM user gets high-energy electronic song",
        user_prefs={"genre": "edm", "mood": "euphoric", "energy": 0.95, "likes_acoustic": False},
        k=3,
        checks=[
            check_result_count(3),
            check_top_genre("edm"),
            check_top_score_above(7.0),
            check_scores_descending(),
        ],
    ),
]


# ── Runner ───────────────────────────────────────────────────────────────────

def run_test(case: TestCase, songs: list) -> tuple[bool, list[str]]:
    """Execute one test case; return (all_passed, list of check messages)."""
    results = recommend_songs(case.user_prefs, songs, k=case.k)
    messages = []
    all_passed = True
    for check in case.checks:
        passed, reason = check(results)
        messages.append(("  ✓" if passed else "  ✗") + f" {reason}")
        if not passed:
            all_passed = False
    return all_passed, messages


def main():
    logger.info("Loading song catalog...")
    try:
        songs = load_songs("data/songs.csv")
    except FileNotFoundError:
        logger.error("data/songs.csv not found. Run from the project root directory.")
        sys.exit(1)

    logger.info(f"Loaded {len(songs)} songs. Running {len(TEST_CASES)} test cases.\n")

    passed_count = 0
    total_count = len(TEST_CASES)
    case_results = []

    for i, case in enumerate(TEST_CASES, 1):
        logger.info(f"[{i}/{total_count}] {case.name}")
        passed, messages = run_test(case, songs)
        case_results.append((case.name, passed, messages))

        status = "PASS" if passed else "FAIL"
        for msg in messages:
            logger.info(msg)
        logger.info(f"  → {status}\n")

        if passed:
            passed_count += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    separator = "=" * 60
    logger.info(separator)
    logger.info("EVALUATION SUMMARY")
    logger.info(separator)
    logger.info(f"Total cases : {total_count}")
    logger.info(f"Passed      : {passed_count}")
    logger.info(f"Failed      : {total_count - passed_count}")
    logger.info(f"Pass rate   : {passed_count / total_count * 100:.1f}%")
    logger.info(separator)

    if passed_count < total_count:
        logger.info("\nFailed cases:")
        for name, passed, messages in case_results:
            if not passed:
                logger.info(f"  - {name}")
                for msg in messages:
                    if "✗" in msg:
                        logger.info(f"    {msg.strip()}")

    logger.info("\nFull results saved to eval_results.log")
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
