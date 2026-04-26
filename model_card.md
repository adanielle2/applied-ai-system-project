# Model Card: RAG Music Recommender

> **Base system:** Music Recommender Simulation (Modules 1–3) — VibeFinder 1.0

---

## 1. Model Name

**VibeFinder 2.0** — RAG-powered music recommendation system

---

## 2. Intended Use

This system suggests songs from an 18-track catalog based on a user's preferred genre, mood, energy level, acoustic preference, and optional free-text description. It is designed for classroom exploration of RAG architecture and is not intended for production use with real users. The primary audience is students and educators studying AI system design.

---

## 3. How It Works

The system combines two recommendation strategies:

**RAG path (when an LLM API key is available):** The user's preferences and any free-text description are turned into a search query, embedded as a vector, and used to retrieve the most semantically similar songs from a ChromaDB vector store. An LLM then reads those candidate songs and writes a ranked list with personalized, natural-language explanations grounded in the retrieved data.

**Fallback path (always available):** If no LLM is configured, the system scores every retrieved song using a weighted formula — genre match, mood match, energy proximity, and acoustic preference — sorts them by score, and returns the top results with scoring reasons. This path runs entirely locally with no API calls.

The key difference from the Module 1–3 version is the retrieval step: instead of scoring the entire catalog every time, the system first narrows the candidate pool to the most semantically relevant songs using vector search, then ranks from that pool.

---

## 4. Data

The catalog has 18 songs covering pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip hop, classical, R&B, metal, folk, EDM, blues, and reggae. Each song has seven numeric and categorical attributes (genre, mood, energy, tempo, valence, danceability, acousticness). The `knowledge_base/song_descriptions.json` file adds one rich paragraph per song describing the sonic texture, use case, and feel — this is what gets embedded for semantic search. The catalog reflects Western popular music genres and does not include K-pop, Latin, Afrobeats, or many other widely listened-to styles.

---

## 5. Strengths

The system works best when the user's preferred genre has multiple catalog entries and when the energy value is specific. Lofi and pop profiles produce clear, intuitive results with high confidence margins. The RAG layer improves on the base system significantly for free-text queries — a user who says "something for a rainy afternoon" now gets relevant results (lofi, folk, ambient) even without specifying a genre label. The fallback scoring ensures the system is always functional, even without an API key. Every recommendation includes a human-readable explanation rather than a black-box ranking.

---

## 6. Limitations and Bias

**Genre dominates the fallback scorer.** A genre match (+1.5 points) still has more influence than a near-perfect energy match in many cases. A song that is a great sonic fit can lose to a weaker genre match.

**Catalog is small and genre-skewed.** With 18 songs, genres like metal, blues, and classical have only one entry each. Users requesting those genres can never receive a full top-5 of their preferred genre.

**RAG quality depends on description quality.** The semantic search is only as good as the natural-language descriptions in the knowledge base. Songs with thin or inaccurate descriptions will rank poorly for relevant queries.

**No memory or personalization.** The system resets every session. It cannot learn that a user consistently skips certain songs or replays others.

**LLM explanations can sound generic.** When the LLM has limited context about a song, its explanation may be plausible-sounding but not particularly specific to why that song fits that user.

**Valence, danceability, and tempo are unused in scoring.** These fields exist in the data but play no role in the ranking formula, meaning users cannot ask for "upbeat" or "danceable" songs directly.

---

## 7. Evaluation

**Unit tests:** 15 pytest tests verified exact score values against the mathematical formula, sort order correctness, acoustic bonuses, and explanation content. 15/15 pass.

**Evaluation harness:** 8 predefined scenarios run against the full system with composable pass/fail checks. 8/8 pass at 100% pass rate.

**Manual profile testing:** Six profiles were tested — three standard (pop, lofi, rock) and three adversarial (unknown genre, conflicting high-energy+sad, perfectly neutral). Standard profiles produced intuitive results with clear score margins. Adversarial profiles exposed the genre-dominance bias: the "conflicting" profile (blues + sad + high energy) returned a slow, quiet blues track ranked first because genre and mood labels matched, even though the energy was completely misaligned. The neutral profile returned results where all five scores were within 0.5 points of each other — effectively random ranking.

---

## 8. Potential Misuse and Safeguards

**Could the system be misused?** A music recommender is lower-risk than many AI systems, but there are still concerns worth naming:

- **Catalog manipulation:** If real artists paid to have their songs added with favorable descriptions, the semantic search could be gamed to surface promoted content ahead of genuinely relevant tracks. Safeguard: keep the catalog and descriptions editorially controlled and versioned.
- **Mood targeting:** A system that knows a user is listening to sad music could theoretically be used to serve them emotionally exploitative content (e.g., advertising targeting vulnerable states). Safeguard: the system should not infer user emotional state or share it with third parties.
- **Proxy discrimination:** Genre preferences can correlate with demographics. A system that over-weights genre could inadvertently produce racially or culturally skewed recommendations. Safeguard: audit recommendations across diverse user profiles and add genre diversity rules.

The current system has no user accounts, no data persistence, and no tracking — by design, every session is stateless, which significantly limits misuse potential.

---

## 9. What Surprised Me During Testing

The most surprising result came from the acoustic-lover evaluation case. I expected an acoustic-preferring user with low energy (0.3) to get a lofi song first, since lofi has both high acousticness and low energy. Instead, the system returned folk first, then ambient, then blues. The reason: the folk song in the catalog has energy 0.31 — nearly identical to the target of 0.3 — while the lofi tracks have energies of 0.35 and 0.40. The energy proximity score is a continuous function that rewards closeness by hundredths of a point, and those small differences added up to more than the acoustic bonus could compensate for.

This surprised me because it revealed that the scoring is more mathematically precise than it "feels" during normal use. The system is not making judgment calls — it is doing arithmetic — and when you look at the actual numbers it behaves exactly as expected. But those expectations are not always what a human would intuitively predict. That gap between intuitive expectation and mathematical behavior is where most of the bias in AI systems lives.

---

## 10. AI Collaboration Reflection

Throughout this project I collaborated closely with an AI assistant (Cursor/Claude) for code architecture, debugging, and documentation.

**One instance where the AI was genuinely helpful:** When I asked how to build the RAG pipeline, the AI suggested using ChromaDB with a local in-memory client and `sentence-transformers` for embedding, rather than relying on OpenAI's embedding API. This was a better design than I had considered — it made the retrieval layer completely free and offline, so the system is functional even without an API key. The fallback architecture (rag_llm → rag_fallback) also came from the AI's suggestion to handle missing API keys gracefully rather than crashing. That pattern made the system meaningfully more reliable.

**One instance where the AI's suggestion was flawed:** When designing the evaluation harness, the AI initially suggested checking whether a specific song title appeared in the top results (e.g., "Library Rain must be #1 for lofi users"). This is a brittle test — it couples the evaluation to the exact current scoring weights, so any legitimate weight adjustment would break the harness even if the system's behavior was still correct. The right approach was to check structural properties (genre of the top result, score ordering, threshold minimums) rather than specific song identity. I caught this issue when I ran the harness and saw that changing the acoustic weight by 0.1 broke three tests that were testing names rather than behavior.

---

## 11. Future Work

The most impactful next step would be replacing the 18-song toy catalog with a real music API (e.g., Spotify's catalog via their API) and generating embeddings for thousands of tracks. A larger catalog would make the semantic search discriminating in ways that are impossible at 18 songs. After that, adding valence and danceability to the scoring formula would let users express preferences like "upbeat" or "danceable" that the current system ignores. A session memory system — even something simple like tracking skipped songs in a local file — would allow the recommendations to improve over time without needing a database.
