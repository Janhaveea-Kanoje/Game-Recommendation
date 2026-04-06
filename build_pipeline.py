"""
build_pipeline.py
─────────────────
Run this script ONCE to generate the pickle files used by the Streamlit app.

Usage:
    python build_pipeline.py --data_dir /path/to/csv/files --out_dir ./

Place all 6 CSV files in data_dir:
    steam.csv
    steam_description_data.csv
    steam_media_data.csv
    steam_requirements_data.csv
    steam_support_info.csv
    steamspy_tag_data.csv
"""
# Import all necessary libraries
import argparse
import os
import ast
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Clean the text by removing tags and special characters and convert
# all to lowercase present in descriptions.

def clean_html(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

# Removes all the semicolons as TF-IDF requires space seperated words and
# not semicolon seperated words

def parse_list_field(field: str) -> str:
    if pd.isna(field):
        return ""
    return " ".join(str(field).replace(";", " ").split())


def parse_screenshots(ss_str) -> list:
    if pd.isna(ss_str):
        return []
    try:
        shots = ast.literal_eval(str(ss_str))
        return [s.get("path_thumbnail", "") for s in shots[:6] if isinstance(s, dict)]
    except Exception:
        return []


def parse_movies(mv_str) -> list:
    if pd.isna(mv_str):
        return []
    try:
        movies = ast.literal_eval(str(mv_str))
        result = []
        for m in movies[:2]:
            if isinstance(m, dict):
                webm = m.get("webm", {})
                mp4 = m.get("mp4", {})
                url = (
                    webm.get("480", "")
                    or mp4.get("480", "")
                    or webm.get("max", "")
                    or mp4.get("max", "")
                )
                if url:
                    result.append({"url": url, "name": m.get("name", "")})
        return result
    except Exception:
        return []

# Load Datasets

def build_pipeline(data_dir: str, out_dir: str, top_n: int = 5000):
    print("─── Loading datasets ───")
    steam = pd.read_csv(os.path.join(data_dir, "steam.csv"))
    desc = pd.read_csv(os.path.join(data_dir, "steam_description_data.csv"))
    media = pd.read_csv(os.path.join(data_dir, "steam_media_data.csv"))
    support = pd.read_csv(os.path.join(data_dir, "steam_support_info.csv"))
    reqs = pd.read_csv(os.path.join(data_dir, "steam_requirements_data.csv"))


    print(f"  steam={steam.shape}, desc={desc.shape}, media={media.shape}")

# Merge Datasets

    df = steam.merge(desc, left_on="appid", right_on="steam_appid", how="left")
    df = df.merge(media, on="steam_appid", how="left")
    df = df.merge(support, on="steam_appid", how="left")
    df = df.merge(reqs[["steam_appid", "minimum", "recommended"]], on="steam_appid", how="left")

    # Filter top rated games for top rated section
    df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]
    df = df.nlargest(top_n, "total_ratings").reset_index(drop=True)
    print(f"  Retained {len(df)} games after filtering")

    # Preprocessing
    print("─── Preprocessing text ───")
    df["clean_desc"] = df["short_description"].apply(clean_html)
    df["clean_about"] = df["about_the_game"].apply(clean_html)
    df["genres_str"] = df["genres"].apply(parse_list_field)
    df["tags_str"] = df["steamspy_tags"].apply(parse_list_field)
    df["categories_str"] = df["categories"].apply(parse_list_field)
    df["developer_str"] = df["developer"].fillna("").str.replace(" ", "_").str.lower()
    df["publisher_str"] = df["publisher"].fillna("").str.replace(" ", "_").str.lower()

    # Create weighted corpus, to assign importance of game feature to
    # determine similarity as not all features have same importance

    df["corpus"] = (
        df["clean_desc"] + " "
        + df["clean_about"] + " "
        + (df["genres_str"] + " ") * 4
        + (df["tags_str"] + " ") * 3
        + df["categories_str"] + " "
        + df["developer_str"] + " "
        + df["publisher_str"]
    )

# TF-IDF Vectorization - Convert text to numbers

    print("─── Building TF-IDF matrix ───")
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(df["corpus"])
    print(f"  TF-IDF shape: {tfidf_matrix.shape}")

    # Cosine Similarity -  Compares each game with all other games and 
    # rank similarity between 0 (no match) to 1 (matched)

    print("─── Computing cosine similarity (chunked) ───")
    chunk = 500
    n = len(df)
    sim = np.zeros((n, n), dtype=np.float16)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        block = cosine_similarity(tfidf_matrix[i:end], tfidf_matrix).astype(np.float16)
        sim[i:end] = block
        print(f"  rows {i}–{end} ✓")
    print("  Similarity matrix complete")

    df["screenshots_list"] = df["screenshots"].apply(parse_screenshots)
    df["movies_list"] = df["movies"].apply(parse_movies)

    df["sentiment_score"] = df["positive_ratings"] / (
        df["positive_ratings"] + df["negative_ratings"] + 1
    )

    # Genre carousel - group games by genre
    genre_map: dict = {}
    for _, row in df.iterrows():
        for g in str(row["genres"]).split(";"):
            g = g.strip()
            if g and g != "nan":
                genre_map.setdefault(g, []).append(int(row["appid"]))

    top_genres = sorted(genre_map.items(), key=lambda x: -len(x[1]))[:12]
    genre_carousel: dict = {}
    for genre, appids in top_genres:
        top12 = df[df["appid"].isin(appids)].nlargest(12, "sentiment_score")
        genre_carousel[genre] = top12["appid"].tolist()

    appid_to_idx = {int(r["appid"]): idx for idx, r in df.iterrows()}

    keep_cols = [
        "appid", "name", "release_date", "developer", "publisher",
        "platforms", "genres", "steamspy_tags", "categories",
        "price", "positive_ratings", "negative_ratings", "sentiment_score",
        "average_playtime", "median_playtime", "owners", "achievements",
        "english", "required_age",
        "short_description", "about_the_game", "detailed_description",
        "header_image", "background", "screenshots_list", "movies_list",
        "website", "minimum", "recommended", "corpus",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df_save = df[existing].copy()

    # Save pickles
    os.makedirs(out_dir, exist_ok=True)
    print(f"─── Saving pickles to {out_dir} ───")

    with open(os.path.join(out_dir, "games_df.pkl"), "wb") as f:
        pickle.dump(df_save, f)
    with open(os.path.join(out_dir, "sim_matrix.pkl"), "wb") as f:
        pickle.dump(sim, f)
    with open(os.path.join(out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(out_dir, "appid_to_idx.pkl"), "wb") as f:
        pickle.dump(appid_to_idx, f)
    with open(os.path.join(out_dir, "genre_carousel.pkl"), "wb") as f:
        pickle.dump(genre_carousel, f)

    print("─── Pipeline complete! ───")
    print(f"  Games: {len(df_save)}")
    print(f"  Genres in carousel: {list(genre_carousel.keys())}")
    print()
    print("Run the Streamlit app:")
    print(f"  streamlit run {os.path.join(out_dir, 'app.py')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SteamLens NLP pipeline")
    parser.add_argument("--data_dir", default=".", help="Directory with CSV files")
    parser.add_argument("--out_dir", default=".", help="Output directory for pickles")
    parser.add_argument("--top_n", type=int, default=5000, help="Max games to index")
    args = parser.parse_args()
    build_pipeline(args.data_dir, args.out_dir, args.top_n)
