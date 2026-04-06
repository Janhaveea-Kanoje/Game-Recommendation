import pandas as pd

def fmt_price(p):
    """
    Format price into readable format
    """
    try:
        p = float(p)
        return "Free" if p == 0 else f"${p:.2f}"
    except:
        return "N/A"

def fmt_number(n):
    """
    Format large numbers into readable format (K, M)
    """
    try:
        n = int(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000: return f"{n/1_000:.0f}K"
        return str(n)
    except:
        return "—"

def rating_label(score):
    """
    Get rating label and color based on sentiment score
    """
    if score >= 0.95: return "Overwhelmingly Positive", "#3fb950"
    if score >= 0.80: return "Very Positive", "#56d364"
    if score >= 0.70: return "Mostly Positive", "#7ee787"
    if score >= 0.40: return "Mixed", "#d29922"
    return "Mostly Negative", "#f85149"

def get_genres(row):
    """
    Extract and clean genres from a dataframe row
    """
    try:
        return [g.strip() for g in str(row.get('genres', '')).split(';') if g.strip() and g.strip() != 'nan'][:3]
    except:
        return []

def img_url(row):
    """
    Get header image URL for a game
    """
    url = row.get('header_image', '')
    if pd.isna(url) or not url:
        return f"https://steamcdn-a.akamaihd.net/steam/apps/{int(row['appid'])}/header.jpg"
    return str(url)
