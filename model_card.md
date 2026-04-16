# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Intended Use

This system suggests songs from a small catalog based on what a user tells it they like. It takes a preferred genre, mood, energy level, and whether they like acoustic music. It is built for classroom exploration — not for real users. It assumes a user can describe their taste with just four inputs, which is a big simplification.

---

## 3. How the Model Works

Each song gets a score based on how well it matches the user. Genre match adds the most points. Mood match adds the second most. Energy is different — instead of just rewarding high or low energy, it rewards songs that are *close* to what the user asked for. There is also a small bonus if the user likes acoustic music and the song fits that. All the scores are added up, the songs are sorted highest to lowest, and the top five are returned. The user also sees a plain-English reason for each pick.

---

## 4. Data

The catalog has 18 songs. It covers pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip hop, classical, R&B, metal, folk, EDM, blues, and reggae. The original starter file had 10 songs — 8 more were added to fill in missing genres and moods. Some genres only have one song (like metal and blues), so the system can never give a full top-5 from those genres alone. The catalog mostly reflects Western popular music and does not include K-pop, Latin, or many other styles that real listeners enjoy.

---

## 5. Strengths

The system works best when the user's favorite genre has multiple songs in the catalog. The lofi and pop profiles produced very clear, sensible results — the right songs rose to the top by a wide margin. The scoring is also easy to explain. Every recommendation comes with a reason, so it never feels like a black box. For a classroom project, that transparency is more useful than a complicated model that nobody can read.

---

## 6. Limitations and Bias

Genre has too much power. A genre match adds more points than a perfect energy match, so a sonically wrong song can beat a sonically right one just because the label matches. The system also ignores valence, danceability, and tempo even though those fields exist in the data. It has no memory — every session starts from scratch. Users who like rare genres (metal, blues, classical) get weak recommendations because those genres are underrepresented. There is also no way to say "I want something new" — the system always picks the closest match, which can feel repetitive.

---

## 7. Evaluation

Six profiles were tested — three normal and three designed to break things. The normal profiles worked great. "Library Rain" and "Midnight Coding" swept the lofi run. "Storm Runner" was almost a perfect match for the rock profile. The weird profiles were more interesting. When the user wanted sad music but high energy, the system picked "Crossroads Lament" — a slow, quiet blues track — just because the genre and mood labels matched. The Perfectly Neutral profile (no genre, no mood, energy 0.5) returned a top 5 where all scores were within 0.5 points of each other, meaning the ranking was basically random. A weight experiment — halving genre, doubling energy — showed the system became much more sensitive to actual sound, which felt like an improvement.

---

## 8. Future Work

The biggest improvement would be adding valence as a scored feature, so users can ask for "upbeat" or "melancholic" music directly. It would also help to track what a user skips or replays so the system can adjust over time. Adding a diversity rule — like "no more than two songs from the same artist or genre in the top 5" — would make results feel less repetitive. A bigger catalog with at least five songs per genre would also make the system much more useful for users with less common tastes.

---

## 9. Personal Reflection

Building this made it clear that a recommender is really just a set of opinions encoded as numbers. Every weight I chose — 3.0 for genre, 2.0 for mood — was a judgment call, and changing those numbers changed who the system "worked for." The most surprising thing was the adversarial profiles. A system that looks perfectly reasonable on normal inputs can behave strangely as soon as you push it outside the cases it was designed for. That made me think differently about apps like Spotify — the recommendations feel smart, but somewhere there are numbers like these deciding what gets shown, and those numbers reflect someone's assumptions about what matters most.
