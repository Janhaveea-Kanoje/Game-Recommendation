import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# set up configuration for page
st.set_page_config(
    page_title="Playing Arena — Game Discovery",
    page_icon="🕹️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS styling
st.markdown("""
<style>
/* ── Root / global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0rem !important; padding-bottom: 2rem; }

/* ── Custom navbar ── */
.navbar {
    background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 100;
}
.navbar-logo {
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, #1db954 0%, #00adb5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.navbar-subtitle {
    color: #8b949e;
    font-size: 13px;
    font-weight: 400;
}

/* ── Hero animations ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulseGlow {
    0%, 100% { text-shadow: 0 0 24px rgba(29,185,84,0.35), 0 0 60px rgba(0,180,216,0.15); }
    50%       { text-shadow: 0 0 48px rgba(0,180,216,0.55), 0 0 100px rgba(29,185,84,0.25); }
}
@keyframes floatIcons {
    0%   { transform: translateY(0px) rotate(0deg);   opacity: 0.06; }
    50%  { transform: translateY(-14px) rotate(4deg); opacity: 0.11; }
    100% { transform: translateY(0px) rotate(0deg);   opacity: 0.06; }
}
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}

/* ── Hero section ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0d1f14 30%, #0a1628 60%, #0d1117 100%);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite;
    padding: 72px 48px 52px;
    text-align: center;
    border-bottom: 1px solid #21262d;
    position: relative;
    overflow: hidden;
}
/* Floating game icons layer */
.hero::before {
    content: "🕹️   🎮   👾   ⚔️   🏆   🎯   🎲   🕹️   🎮   👾";
    position: absolute;
    top: 10px; left: -20px; right: -20px;
    font-size: 28px;
    letter-spacing: 40px;
    pointer-events: none;
    animation: floatIcons 7s ease-in-out infinite;
    white-space: nowrap;
}
/* Subtle scanline shimmer */
.hero::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(29,185,84,0.4), transparent);
    animation: scanline 4s linear infinite;
    pointer-events: none;
}
.hero-title-wrap {
    animation: fadeSlideDown 0.8s cubic-bezier(0.22,1,0.36,1) both;
    position: relative;
    z-index: 1;
}
.hero h1 {
    font-size: 52px;
    font-weight: 800;
    color: #e6edf3;
    margin: 0 0 8px;
    line-height: 1.15;
    letter-spacing: -1px;
}
.hero h1 span {
    background: linear-gradient(135deg, #1db954 0%, #00b4d8 50%, #1db954 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite, pulseGlow 3s ease-in-out infinite;
}
.hero p {
    color: #8b949e;
    font-size: 17px;
    max-width: 560px;
    margin: 0 auto 28px;
    font-weight: 400;
    line-height: 1.65;
}

/* ── Genre pill ── */
.genre-pill {
    display: inline-block;
    background: #21262d;
    color: #58a6ff;
    border: 1px solid #30363d;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
    cursor: pointer;
    transition: all 0.2s;
}
.genre-pill:hover {
    background: #1f6feb;
    color: white;
    border-color: #1f6feb;
}

/* ── Section headings ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 36px 0 16px;
    padding: 0 2px;
}
.section-header h2 {
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
    margin: 0;
}
.section-accent {
    width: 4px;
    height: 22px;
    border-radius: 2px;
    background: linear-gradient(180deg, #1db954, #00b4d8);
}

/* ── Game card ── */
.game-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
    cursor: pointer;
    height: 100%;
}
.game-card:hover {
    border-color: #1f6feb;
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(31,111,235,0.18);
}
.game-card img {
    width: 100%;
    aspect-ratio: 460/215;
    object-fit: cover;
}
.game-card-body {
    padding: 12px 14px;
}
.game-card-title {
    font-size: 13.5px;
    font-weight: 600;
    color: #e6edf3;
    margin: 0 0 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.game-card-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
}
.game-card-genre {
    font-size: 11px;
    color: #58a6ff;
    background: rgba(31,111,235,0.12);
    padding: 2px 8px;
    border-radius: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 110px;
}
.game-card-price {
    font-size: 12px;
    font-weight: 700;
    color: #3fb950;
}
.game-card-rating {
    font-size: 11px;
    color: #f0c27f;
}

/* ── Clickable thumbnail hover card ── */
.thumb-wrap {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    display: block;
    border: 2px solid transparent;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.thumb-wrap:hover {
    border-color: #1f6feb;
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(31,111,235,0.30);
}
.thumb-wrap img {
    width: 100%;
    display: block;
    aspect-ratio: 460/215;
    object-fit: cover;
    transition: filter 0.25s;
}
.thumb-wrap:hover img {
    filter: brightness(0.35);
}
.thumb-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    opacity: 0;
    transition: opacity 0.25s;
    padding: 10px;
    text-align: center;
}
.thumb-wrap:hover .thumb-overlay {
    opacity: 1;
}
.thumb-overlay-name {
    font-size: 13px;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.3;
    text-shadow: 0 1px 4px rgba(0,0,0,0.9);
}
.thumb-overlay-price {
    font-size: 15px;
    font-weight: 800;
    color: #3fb950;
    text-shadow: 0 1px 4px rgba(0,0,0,0.9);
}
.thumb-overlay-cta {
    margin-top: 4px;
    background: #1f6feb;
    color: white;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 0.4px;
}
/* Hide the Streamlit nav buttons used by JS click */
.hidden-nav-btn > div > button {
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

/* ── Detail page ── */
.detail-hero {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 28px;
}
.detail-hero-bg {
    width: 100%;
    max-height: 340px;
    object-fit: cover;
    display: block;
    filter: brightness(1);
}
.detail-hero-overlay {
    position: absolute;
    bottom: 0;
    left: 0; right: 0;
    padding: 28px 32px;
    background: linear-gradient(0deg, rgba(13,17,23,0.98) 0%, transparent 100%);
}
.detail-title {
    font-size: 36px;
    font-weight: 800;
    color: #e6edf3;
    margin: 0 0 8px;
    line-height: 1.15;
}
.detail-developer {
    font-size: 14px;
    color: #8b949e;
    margin: 0;
}

/* Info badges */
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
.badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
    color: #c9d1d9;
    display: flex;
    align-items: center;
    gap: 5px;
}
.badge.green { border-color: #3fb950; color: #3fb950; background: rgba(63,185,80,0.08); }
.badge.blue  { border-color: #58a6ff; color: #58a6ff; background: rgba(88,166,255,0.08); }
.badge.yellow{ border-color: #d29922; color: #d29922; background: rgba(210,153,34,0.08); }

/* Screenshots gallery */
.screenshot-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 16px 0;
}
.screenshot-grid img {
    width: 100%;
    border-radius: 8px;
    border: 1px solid #21262d;
}

/* Rec cards */
.rec-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.2s;
}
.rec-card:hover {
    border-color: #58a6ff;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(88,166,255,0.12);
}
.rec-card img {
    width: 100%;
    aspect-ratio: 460/215;
    object-fit: cover;
}
.rec-card-body { padding: 10px 12px; }
.rec-card-title {
    font-size: 12.5px;
    font-weight: 600;
    color: #e6edf3;
    margin: 0 0 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.rec-score {
    font-size: 11px;
    color: #58a6ff;
}

/* Search results */
.search-result {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.15s;
}
.search-result:hover {
    border-color: #1f6feb;
    background: #1c2128;
}
.search-result img {
    width: 80px;
    border-radius: 5px;
    flex-shrink: 0;
}
.search-result-info h4 {
    font-size: 14px;
    font-weight: 600;
    color: #e6edf3;
    margin: 0 0 3px;
}
.search-result-info p {
    font-size: 12px;
    color: #8b949e;
    margin: 0;
}

/* Divider */
.steam-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 32px 0;
}

/* Rating bar */
.rating-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-top: 4px;
}
.rating-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #1db954, #3fb950);
}

/* Stat box */
.stat-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-box .stat-value {
    font-size: 24px;
    font-weight: 800;
    color: #e6edf3;
}
.stat-box .stat-label {
    font-size: 11px;
    color: #8b949e;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Back button */
.back-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #8b949e;
    font-size: 13px;
    cursor: pointer;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #21262d;
    background: #161b22;
    transition: all 0.15s;
    margin: 16px 0;
}
.back-btn:hover { color: #e6edf3; border-color: #30363d; }

/* Horizontal scroll carousel */
.carousel-scroll {
    display: flex;
    overflow-x: auto;
    gap: 14px;
    padding-bottom: 8px;
    scrollbar-width: thin;
    scrollbar-color: #21262d transparent;
}
.carousel-scroll::-webkit-scrollbar { height: 5px; }
.carousel-scroll::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }

.carousel-item {
    flex: 0 0 200px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.2s;
}
.carousel-item:hover {
    border-color: #58a6ff;
    transform: translateY(-3px);
}
.carousel-item img {
    width: 100%;
    aspect-ratio: 460/215;
    object-fit: cover;
}
.carousel-item-body { padding: 8px 10px; }
.carousel-item-title {
    font-size: 12px;
    font-weight: 600;
    color: #e6edf3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin: 0 0 3px;
}
.carousel-item-price {
    font-size: 11px;
    color: #3fb950;
    font-weight: 700;
}

/* Active genre tab */
.genre-tab-active {
    background: #1f6feb !important;
    color: white !important;
    border-color: #1f6feb !important;
}

/* Footer */
.app-footer {
    border-top: 1px solid #21262d;
    padding: 24px 48px;
    color: #484f58;
    font-size: 12px;
    text-align: center;
    margin-top: 48px;
}
</style>
""", unsafe_allow_html=True)

# Load Data and direct path
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner="Loading Playing Arena engine...")
def load_data():
    with open(DATA_DIR / 'games_df.pkl', 'rb') as f:
        df = pickle.load(f)
    with open(DATA_DIR / 'sim_matrix.pkl', 'rb') as f:
        sim = pickle.load(f)
    with open(DATA_DIR / 'tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open(DATA_DIR / 'appid_to_idx.pkl', 'rb') as f:
        idx_map = pickle.load(f)
    with open(DATA_DIR / 'genre_carousel.pkl', 'rb') as f:
        genre_carousel = pickle.load(f)
    return df, sim, tfidf, idx_map, genre_carousel

df, sim_matrix, tfidf_vec, appid_to_idx, genre_carousel = load_data()

# Get api ID, to get games recommendations, images, generes etc.
def get_game(appid):
    appid = int(appid)
    if appid not in appid_to_idx:
        return None
    return df.iloc[appid_to_idx[appid]]

def get_recommendations(appid, n=12):
    appid = int(appid)
    if appid not in appid_to_idx:
        return []
    idx = appid_to_idx[appid]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: -float(x[1]))
    scores = [s for s in scores if s[0] != idx][:n]

    # Rescale similarity scores so the best match shows ~92% instead of raw ~40%
    # Raw TF-IDF cosine similarity rarely exceeds 0.5 between different games.
    # We rescale: map [0, top_score] → [0, 0.92] so percentages are meaningful.
    top_score = float(scores[0][1]) if scores else 1.0
    scale = 0.92 / top_score if top_score > 0 else 1.0

    results = []
    for i, score in scores:
        row = df.iloc[i]
        rescaled = min(float(score) * scale, 0.99)  # cap at 99%
        results.append({
            'appid': int(row['appid']),
            'name': row['name'],
            'header_image': row.get('header_image', ''),
            'genres': row.get('genres', ''),
            'price': row.get('price', 0),
            'score': rescaled,
            'sentiment': row.get('sentiment_score', 0),
        })
    return results

# Perform semantic analysis, and cosin similarity to understand the text and convert them into vectors, and then to determine 
# the similarity between 2 vectors/games.

def semantic_search(query, n=20):
    query_vec = tfidf_vec.transform([query.lower()])
    scores = cosine_similarity(query_vec, tfidf_vec.transform(df['corpus'].fillna(''))).flatten()
    top_idx = scores.argsort()[::-1][:n]
    results = []
    for i in top_idx:
        if scores[i] > 0.01:
            row = df.iloc[i]
            results.append({
                'appid': int(row['appid']),
                'name': row['name'],
                'header_image': row.get('header_image', ''),
                'genres': row.get('genres', ''),
                'price': row.get('price', 0),
                'score': float(scores[i]),
                'short_description': str(row.get('short_description', ''))[:120],
            })
    return results

def fmt_price(p):
    try:
        p = float(p)
        return "Free" if p == 0 else f"${p:.2f}"
    except:
        return "N/A"

def fmt_number(n):
    try:
        n = int(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000: return f"{n/1_000:.0f}K"
        return str(n)
    except:
        return "—"

def rating_label(score):
    if score >= 0.95: return "Overwhelmingly Positive", "#3fb950"
    if score >= 0.80: return "Very Positive", "#56d364"
    if score >= 0.70: return "Mostly Positive", "#7ee787"
    if score >= 0.40: return "Mixed", "#d29922"
    return "Mostly Negative", "#f85149"

def get_genres(row):
    try:
        return [g.strip() for g in str(row.get('genres', '')).split(';') if g.strip() and g.strip() != 'nan'][:3]
    except:
        return []

def img_url(row):
    url = row.get('header_image', '')
    if pd.isna(url) or not url:
        return f"https://steamcdn-a.akamaihd.net/steam/apps/{int(row['appid'])}/header.jpg"
    return str(url)

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_appid' not in st.session_state:
    st.session_state.selected_appid = None
if 'active_genre' not in st.session_state:
    st.session_state.active_genre = list(genre_carousel.keys())[0]
if 'search_query' not in st.session_state:
    st.session_state.search_query = ''

def go_to_game(appid):
    st.session_state.selected_appid = int(appid)
    st.session_state.page = 'detail'
    st.rerun()

def go_home():
    st.session_state.page = 'home'
    st.session_state.selected_appid = None
    st.rerun()

# Design Navigation Bar
st.markdown("""
<div class="navbar">
  <span class="navbar-logo" id="nav-logo-click">🕹️ Playing Arena</span>
  <span class="navbar-subtitle">AI-Powered Game Discovery</span>
</div>
<style>
#nav-logo-click { cursor: pointer; }
</style>
<script>
(function waitForLogo() {
    var logo = window.parent.document.getElementById('nav-logo-click');
    if (!logo) { setTimeout(waitForLogo, 200); return; }
    logo.addEventListener('click', function() {
        var url = new URL(window.parent.location.href);
        url.searchParams.set('nav', 'home');
        window.parent.history.pushState({}, '', url);
        window.parent.dispatchEvent(new Event('popstate'));
    });
})();
</script>
""", unsafe_allow_html=True)

if st.query_params.get('nav') == 'home':
    st.query_params.clear()
    go_home()


# Design Home Page

if st.session_state.page == 'home':

    st.markdown("""
    <div class="hero">
      <div class="hero-title-wrap">
        <h1>Welcome to<br><span>Playing Arena</span></h1>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns([1, 3, 1])
    with col_s2:
        search_input = st.text_input(
            label="search",
            placeholder="🔍  Search games, genres, tags, or describe what you want...",
            label_visibility="collapsed",
            key="main_search"
        )

    if search_input and len(search_input) >= 2:
        results = semantic_search(search_input, n=15)
        if results:
            st.markdown(f"""
            <div class="section-header">
              <div class="section-accent"></div>
              <h2>Search Results for "{search_input}"</h2>
            </div>""", unsafe_allow_html=True)

            for r in results[:8]:
                cols = st.columns([1, 6, 1])
                with cols[0]:
                    st.image(r['header_image'], use_container_width=True)
                with cols[1]:
                    genres_disp = ' · '.join(r['genres'].split(';')[:2]) if r.get('genres') else ''
                    st.markdown(f"""
                    <div>
                      <div style="font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:3px">{r['name']}</div>
                      <div style="font-size:12px;color:#58a6ff;margin-bottom:5px">{genres_disp}</div>
                      <div style="font-size:12px;color:#8b949e">{r['short_description'][:160]}...</div>
                    </div>""", unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(f"<div style='font-size:13px;font-weight:700;color:#3fb950;text-align:right;margin-top:8px'>{fmt_price(r['price'])}</div>", unsafe_allow_html=True)
                    if st.button("View →", key=f"sr_{r['appid']}"):
                        go_to_game(r['appid'])
                st.markdown("<hr style='border-color:#21262d;margin:4px 0'>", unsafe_allow_html=True)
        else:
            st.info("No results found. Try different keywords.")

    else:
        # Design carousel for different genres
        genres = list(genre_carousel.keys())

        st.markdown("""
        <div class="section-header">
          <div class="section-accent"></div>
          <h2>Browse by Genre</h2>
        </div>""", unsafe_allow_html=True)

        # Dropdown for different genre categories
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            selected_genre = st.selectbox(
                "Select a Genre",
                options=genres,
                index=genres.index(st.session_state.active_genre) if st.session_state.active_genre in genres else 0,
                key="genre_select",
                label_visibility="collapsed",
            )
            if selected_genre != st.session_state.active_genre:
                st.session_state.active_genre = selected_genre
                st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        active_genre = st.session_state.active_genre
        appids = genre_carousel.get(active_genre, [])
        genre_games = [get_game(a) for a in appids[:12] if get_game(a) is not None]

        if genre_games:
           
            for row_start in range(0, min(12, len(genre_games)), 4):
                row_games = genre_games[row_start:row_start+4]
                cols = st.columns(4)
                for col_idx, game in enumerate(row_games):
                    with cols[col_idx]:
                        genres_g = get_genres(game)
                        genre_label = genres_g[0] if genres_g else active_genre
                        score = float(game.get('sentiment_score', 0))
                        label, _ = rating_label(score)
                        price_disp = fmt_price(game.get('price', 0))
                        appid_g = int(game['appid'])
                        nav_label = f"__nav_{appid_g}__{row_start}_{col_idx}"
                        safe_name = str(game['name']).replace('"', '&quot;').replace("'", "&#39;")

                        st.markdown(f"""
                        <div class="thumb-wrap" onclick="
                          var btns = window.parent.document.querySelectorAll('button');
                          for(var b of btns){{
                            if(b.innerText.trim() === '{nav_label}'){{b.click();break;}}
                          }}
                        ">
                          <img src="{img_url(game)}" alt="{safe_name}"/>
                          <div class="thumb-overlay">
                            <div class="thumb-overlay-name">{safe_name}</div>
                            <div class="thumb-overlay-price">{price_disp}</div>
                            <div class="thumb-overlay-cta">&#9654; View Game</div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="margin-top:6px">
                          <div style="font-size:13px;font-weight:600;color:#e6edf3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{game['name']}</div>
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px">
                            <span style="font-size:11px;color:#58a6ff;background:rgba(88,166,255,0.1);padding:2px 8px;border-radius:10px">{genre_label}</span>
                            <span style="font-size:12px;font-weight:700;color:#3fb950">{price_disp}</span>
                          </div>
                          <div style="font-size:11px;color:#8b949e;margin-top:3px">⭐ {label}</div>
                        </div>""", unsafe_allow_html=True)

                        st.markdown('<div class="hidden-nav-btn">', unsafe_allow_html=True)
                        if st.button(nav_label, key=f"card_{appid_g}_{row_start}_{col_idx}"):
                            go_to_game(appid_g)
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Grid for top rated section
        st.markdown("<hr class='steam-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
          <div class="section-accent"></div>
          <h2>🏆 Top Rated on Steam</h2>
        </div>""", unsafe_allow_html=True)

        top_rated = df.nlargest(8, 'sentiment_score')
        tr_cols = st.columns(4)
        for i, (_, game) in enumerate(top_rated.iterrows()):
            with tr_cols[i % 4]:
                genres_g = get_genres(game)
                score = float(game.get('sentiment_score', 0))
                lbl, clr = rating_label(score)
                price_disp = fmt_price(game.get('price', 0))
                appid_tr = int(game['appid'])
                nav_label_tr = f"__navtr_{appid_tr}__{i}"
                safe_name_tr = str(game['name']).replace('"', '&quot;').replace("'", "&#39;")

                st.markdown(f"""
                <div class="thumb-wrap" onclick="
                  var btns = window.parent.document.querySelectorAll('button');
                  for(var b of btns){{
                    if(b.innerText.trim() === '{nav_label_tr}'){{b.click();break;}}
                  }}
                ">
                  <img src="{img_url(game)}" alt="{safe_name_tr}"/>
                  <div class="thumb-overlay">
                    <div class="thumb-overlay-name">{safe_name_tr}</div>
                    <div class="thumb-overlay-price">{price_disp}</div>
                    <div class="thumb-overlay-cta">&#9654; View Game</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:4px">
                  <div style="font-size:13px;font-weight:600;color:#e6edf3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{game['name']}</div>
                  <div style="font-size:11px;color:{clr};margin-top:3px">★ {lbl}</div>
                  <div style="font-size:12px;font-weight:700;color:#3fb950;margin-top:2px">{price_disp}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="hidden-nav-btn">', unsafe_allow_html=True)
                if st.button(nav_label_tr, key=f"tr_{appid_tr}_{i}"):
                    go_to_game(appid_tr)
                st.markdown('</div>', unsafe_allow_html=True)

            if i == 3:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("<hr class='steam-divider'>", unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown("""<div class="stat-box"><div class="stat-value">5,000+</div><div class="stat-label">Games Indexed</div></div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown("""<div class="stat-box"><div class="stat-value">12</div><div class="stat-label">Genres Covered</div></div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown("""<div class="stat-box"><div class="stat-value">NLP</div><div class="stat-label">Semantic Search</div></div>""", unsafe_allow_html=True)
        with sc4:
            st.markdown("""<div class="stat-box"><div class="stat-value">TF-IDF</div><div class="stat-label">Similarity Engine</div></div>""", unsafe_allow_html=True)

# Page 2 - Details page Design anf functionality
elif st.session_state.page == 'detail':
    st.markdown("""
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)
    appid = st.session_state.selected_appid
    game = get_game(appid)

    if game is None:
        st.error("Game not found.")
        if st.button("← Back to Home"):
            go_home()
        st.stop()

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_btn"):
        go_home()

    bg = game.get('background', '')
    header = img_url(game)
    banner_url = header if header else (str(bg) if bg and not pd.isna(bg) and str(bg) != 'nan' else header)

    genres_list = get_genres(game)
    score = float(game.get('sentiment_score', 0))
    rating_lbl, rating_clr = rating_label(score)

    col_banner, col_info = st.columns([3, 2])
    with col_banner:
        st.image(banner_url, use_container_width=True)
    with col_info:
        st.markdown(f"""
        <div style="padding:8px 0">
          <h1 style="font-size:28px;font-weight:800;color:#e6edf3;margin:0 0 8px;line-height:1.2">{game['name']}</h1>
          <div style="font-size:14px;color:#8b949e;margin-bottom:14px">
            by <span style="color:#58a6ff">{game.get('developer','Unknown')}</span>
            · {game.get('publisher','')}
          </div>
        """, unsafe_allow_html=True)

        for g in genres_list:
            st.markdown(f'<span class="genre-pill">{g}</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:16px">
          <div style="font-size:13px;color:#8b949e;margin-bottom:4px">Overall Rating</div>
          <div style="font-size:15px;font-weight:700;color:{rating_clr}">{rating_lbl}</div>
          <div class="rating-bar-bg" style="width:200px">
            <div class="rating-bar-fill" style="width:{int(score*100)}%"></div>
          </div>
          <div style="font-size:12px;color:#8b949e;margin-top:4px">
            {fmt_number(game.get('positive_ratings',0))} positive · {fmt_number(game.get('negative_ratings',0))} negative
          </div>
        </div>
        """, unsafe_allow_html=True)

        price_val = game.get('price', 0)
        price_display = fmt_price(price_val)
        st.markdown(f"""
        <div style="margin-top:20px;padding:16px;background:#161b22;border:1px solid #21262d;border-radius:10px;display:inline-block">
          <div style="font-size:28px;font-weight:800;color:#3fb950">{price_display}</div>
          <div style="font-size:12px;color:#8b949e;margin-top:2px">Released {game.get('release_date','N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        owners = str(game.get('owners', 'N/A'))
        st.markdown(f"""<div class="stat-box"><div class="stat-value" style="font-size:17px">{owners}</div><div class="stat-label">Owners</div></div>""", unsafe_allow_html=True)
    with s2:
        avgt = int(game.get('average_playtime', 0))
        st.markdown(f"""<div class="stat-box"><div class="stat-value">{avgt}h</div><div class="stat-label">Avg Playtime</div></div>""", unsafe_allow_html=True)
    with s3:
        ach = int(game.get('achievements', 0))
        st.markdown(f"""<div class="stat-box"><div class="stat-value">{ach}</div><div class="stat-label">Achievements</div></div>""", unsafe_allow_html=True)
    with s4:
        platforms = str(game.get('platforms', '')).replace(';', ' · ').title()
        st.markdown(f"""<div class="stat-box"><div class="stat-value">{platforms}</div><div class="stat-label">Platforms</div></div>""", unsafe_allow_html=True)

    st.markdown("<hr class='steam-divider'>", unsafe_allow_html=True)

    col_desc, col_sidebar = st.columns([3, 1])
    with col_desc:
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>About This Game</h2></div>""", unsafe_allow_html=True)
        about = game.get('short_description', '')
        if pd.isna(about) or not about:
            about = game.get('about_the_game', '')
        if not pd.isna(about) and about:
            clean = re.sub(r'<[^>]+>', '', str(about))
            st.markdown(f"<p style='color:#c9d1d9;line-height:1.75;font-size:14px'>{clean}</p>", unsafe_allow_html=True)

        tags_raw = str(game.get('steamspy_tags', ''))
        tags_list = [t.strip() for t in tags_raw.split(';') if t.strip() and t.strip() != 'nan']
        if tags_list:
            st.markdown("""<div class="section-header" style="margin-top:24px"><div class="section-accent"></div><h2>Tags</h2></div>""", unsafe_allow_html=True)
            tags_html = ''.join([f'<span class="genre-pill">{t}</span>' for t in tags_list[:20]])
            st.markdown(tags_html, unsafe_allow_html=True)

    with col_sidebar:
        st.markdown("""<div style="margin-top:40px">""", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:13px;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px">Game Details</div>""", unsafe_allow_html=True)

        details = [
            ("📅 Release Date", game.get('release_date', 'N/A')),
            ("🏷️ Genre", ', '.join(genres_list) or 'N/A'),
            ("💰 Price", fmt_price(game.get('price', 0))),
            ("🏆 Achievements", str(int(game.get('achievements', 0)))),
            ("🌐 Languages", "English" if game.get('english', 0) else "Multiple"),
        ]
        for label, val in details:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #21262d">
              <span style="font-size:12px;color:#8b949e">{label}</span>
              <span style="font-size:12px;color:#e6edf3;font-weight:500;text-align:right;max-width:120px">{val}</span>
            </div>""", unsafe_allow_html=True)

        website = game.get('website', '')
        if website and not pd.isna(website):
            st.markdown(f"""<div style="margin-top:12px"><a href="{website}" target="_blank" style="color:#58a6ff;font-size:12px">🌐 Official Website</a></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Scrrenshots display for each selected game
    screenshots = game.get('screenshots_list', [])
    if isinstance(screenshots, list) and len(screenshots) > 0:
        st.markdown("<hr class='steam-divider'>", unsafe_allow_html=True)
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>📸 Screenshots</h2></div>""", unsafe_allow_html=True)
        valid_shots = [s for s in screenshots if s and str(s) != 'nan'][:6]
        if valid_shots:
            shot_cols = st.columns(min(3, len(valid_shots)))
            for i, shot in enumerate(valid_shots):
                with shot_cols[i % 3]:
                    st.image(shot, use_container_width=True)

    minimum = game.get('minimum', '')
    if minimum and not pd.isna(minimum) and str(minimum) != 'nan':
        st.markdown("<hr class='steam-divider'>", unsafe_allow_html=True)
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>💻 System Requirements</h2></div>""", unsafe_allow_html=True)
        req_col1, req_col2 = st.columns(2)
        with req_col1:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px">
              <div style="font-size:13px;font-weight:700;color:#58a6ff;margin-bottom:10px">MINIMUM</div>
              <div style="font-size:12px;color:#8b949e;line-height:1.8">{re.sub(r'<[^>]+>',' ',str(minimum))}</div>
            </div>""", unsafe_allow_html=True)
        recommended = game.get('recommended', '')
        if recommended and not pd.isna(recommended) and str(recommended) != 'nan':
            with req_col2:
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px">
                  <div style="font-size:13px;font-weight:700;color:#3fb950;margin-bottom:10px">RECOMMENDED</div>
                  <div style="font-size:12px;color:#8b949e;line-height:1.8">{re.sub(r'<[^>]+>',' ',str(recommended))}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='steam-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <div class="section-accent"></div>
      <h2>🤖 Similar Games You May Like</h2>
    </div>
    <p style="color:#8b949e;font-size:13px;margin-top:-8px;margin-bottom:20px">
      Powered by NLP semantic similarity · Cosine distance on TF-IDF corpus
    </p>""", unsafe_allow_html=True)

    recs = get_recommendations(appid, n=12)
    if recs:
        for row_start in range(0, min(12, len(recs)), 4):
            row_recs = recs[row_start:row_start+4]
            rec_cols = st.columns(4)
            for ci, rec in enumerate(row_recs):
                with rec_cols[ci]:
                    genre_r = rec['genres'].split(';')[0].strip() if rec['genres'] else ''
                    sim_pct = int(rec['score'] * 100)
                    lbl_r, clr_r = rating_label(rec['sentiment'])
                    st.image(rec['header_image'] or f"https://steamcdn-a.akamaihd.net/steam/apps/{rec['appid']}/header.jpg",
                             use_container_width=True)
                    st.markdown(f"""
                    <div style="margin-top:4px">
                      <div style="font-size:12.5px;font-weight:600;color:#e6edf3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{rec['name']}</div>
                      <div style="display:flex;justify-content:space-between;margin-top:4px">
                        <span style="font-size:11px;color:#58a6ff">{genre_r}</span>
                        <span style="font-size:11px;color:#8b949e">{sim_pct}% match</span>
                      </div>
                      <div style="font-size:12px;font-weight:700;color:#3fb950;margin-top:3px">{fmt_price(rec['price'])}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("View →", key=f"rec_{rec['appid']}_{row_start}_{ci}", use_container_width=True):
                        go_to_game(rec['appid'])
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
