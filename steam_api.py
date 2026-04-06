import requests
import streamlit as st

class SteamAPI:
    """Handler for Steam API operations (Placeholder for consistency with sample)"""
    
    def __init__(self):
        """Initialize Steam API handler"""
        self.base_url = "https://store.steampowered.com/api/appdetails"
    
    def get_game_details(self, appid):
        """
        Get specific game details from Steam API
        
        Args:
            appid (int): Steam App ID
            
        Returns:
            dict: Game data or None
        """
        try:
            params = {"appids": appid}
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if data and str(appid) in data and data[str(appid)]['success']:
                return data[str(appid)]['data']
            return None
        except Exception as e:
            st.error(f"Error fetching Steam details: {e}")
            return None

    def get_header_image(self, appid):
        """
        Get header image URL for a game
        """
        return f"https://steamcdn-a.akamaihd.net/steam/apps/{appid}/header.jpg"
