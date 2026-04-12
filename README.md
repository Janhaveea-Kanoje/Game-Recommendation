# 🎮 Playing Arena — Smart Game Recommendation System

## 📌 Overview

**Playing Arena** is a content-based game recommendation system built with Streamlit. It uses TF-IDF vectorisation and cosine similarity on Steam game descriptions, genres, and user tags to find games that are truly similar to what you like — not just what's trending.

🔗 **Live App:** [playing-arena.streamlit.app](https://game-recommendation-6tfd7ykkgfbjmvdtmrjfk5.streamlit.app/)

---

## ✨ Features

- 🔍 **Semantic Search** — Search by title, genre, tags, or describe what you want in plain English
- 🎯 **AI Recommendations** — Get 12 similar games powered by TF-IDF + cosine similarity
- 🎮 **Genre Browsing** — Horizontal scrollable genre pill strip + dropdown, both in sync
- 🖼️ **Hover Cards** — Hover over any thumbnail to preview game name and price
- 📄 **Game Detail Pages** — Full info including ratings, playtime, screenshots, system requirements
- 🏆 **Top Rated Section** — Browse the highest-rated games on Steam
- ⚡ **Fast Loading** — Precomputed similarity matrix for instant recommendations

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS/HTML |
| NLP Engine | scikit-learn TF-IDF Vectorizer |
| Similarity | Cosine Similarity (precomputed matrix) |
| Data | Steam dataset — 5,000 games |
| Storage | Google Drive (pickle files via gdown) |
| Deployment | Streamlit Cloud + GitHub |

---

## 📁 Project Structure

```
playing-arena/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes pkl files from git
└── README.md               # This file

# The following pkl files are NOT in the repo (stored on Google Drive)
# games_df.pkl              # Game metadata DataFrame
# sim_matrix.pkl            # 5000x5000 cosine similarity matrix
# tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
# appid_to_idx.pkl          # App ID to DataFrame index map
# genre_carousel.pkl        # Games grouped by genre
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/playing-arena.git
cd playing-arena
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Google Drive File IDs

Open `app.py` and fill in your file IDs in the `GDRIVE_FILE_IDS` dictionary:

```python
GDRIVE_FILE_IDS = {
    "games_df.pkl":         "your_file_id_here",
    "sim_matrix.pkl":       "your_file_id_here",
    "tfidf_vectorizer.pkl": "your_file_id_here",
    "appid_to_idx.pkl":     "your_file_id_here",
    "genre_carousel.pkl":   "your_file_id_here",
}
```

### 4. Run locally

```bash
streamlit run app.py
```

---

## 🧠 How the Recommendation Engine Works

```
Game Corpus (name + genres + tags + description)
            ↓
    TF-IDF Vectorisation
    (5000 games × ~9500 terms)
            ↓
   Cosine Similarity Matrix
        (5000 × 5000)
            ↓
   Top 12 Similar Games
   returned instantly on click
```

1. **Corpus Construction** — Each game's name, genres (weighted 3×), steamspy tags (weighted 2×), and short description are combined into a single text field
2. **TF-IDF** — Converts each corpus into a sparse vector. Rare, specific terms (like *Roguelike* or *Metroidvania*) carry more weight than common words
3. **Cosine Similarity** — Measures the angle between two game vectors. Scores closer to 1 mean highly similar games
4. **Precomputed Matrix** — All similarities are calculated once offline and saved, so recommendations are instant at runtime

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
gdown>=4.7.1
```

---

## ☁️ Deployment on Streamlit Cloud

1. Push `app.py`, `requirements.txt`, and `.gitignore` to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `app.py`
4. Click **Deploy**

> ⚠️ The pkl files are downloaded automatically from Google Drive on first load (~3-5 mins). Subsequent loads are instant due to Streamlit caching.

---

## 📊 Dataset

- **Source:** Steam Games Dataset (Kaggle / Steam Spy)
- **Size:** 5,000 games
- **Fields:** name, genres, steamspy_tags, short_description, price, sentiment_score, positive_ratings, negative_ratings, owners, average_playtime, achievements, platforms, release_date, header_image, screenshots

---

## 🔮 Future Improvements

- [ ] User accounts and personalised recommendation history
- [ ] Collaborative filtering based on user behaviour
- [ ] Live Steam API pricing integration
- [ ] Mood-based and tempo filters
- [ ] Mobile app (React Native)

---

## 👩‍💻 Student Details

**Janhavee Kanoje**
Student ID: GH1033705
Module: Business Project in Big Data & AI
Gisma University of Applied Sciences
