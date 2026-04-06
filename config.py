"""
Configuration file for Playing Arena — Game Discovery
"""

import os

# App Configuration
APP_TITLE = "Playing Arena — Game Discovery"
APP_ICON = "🕹️"
PAGE_LAYOUT = "wide"

# Search Configuration
DEFAULT_SEARCH_LIMIT = 20
RECOMMENDATIONS_LIMIT = 12

# UI Configuration
GRID_COLUMNS = 4

# Colors
PRIMARY_COLOR = "#1DB954"
SECONDARY_COLOR = "#00adb5"
BACKGROUND_COLOR = "#0d1117"

# Model Configuration
MODEL_DF_PATH = "games_df.pkl"
MODEL_SIM_PATH = "sim_matrix.pkl"
MODEL_TFIDF_PATH = "tfidf_vectorizer.pkl"
MODEL_IDX_MAP_PATH = "appid_to_idx.pkl"
MODEL_GENRE_CAROUSEL_PATH = "genre_carousel.pkl"

# Error Messages
ERROR_MODEL_NOT_FOUND = "❌ Model files not found! Please ensure all .pkl files are in the directory."
ERROR_NO_RESULTS = "No results found. Try a different search term."
