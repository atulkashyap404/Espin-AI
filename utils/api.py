import requests
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

@st.cache_data(ttl=300, show_spinner=False)  # Cache API responses for 5 minutes
def fetch_groq_response(user_prompt):
    """Fetch response from Groq API with structured messages and optimized handling."""
    
    if not GROQ_API_KEY:
        return "❌ Error: API key not found. Please set the GROQ_API_KEY environment variable."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Chat message structure
    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in nanofiber research. Provide accurate, well-structured responses."},
        {"role": "user", "content": user_prompt}
    ]

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 500,  # Ensure longer responses
        "temperature": 0.7,  # Balanced creativity
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }

    try:
        with st.spinner():  # ✅ Keeps spinner but hides extra text
            response = requests.post(GROQ_ENDPOINT, json=data, headers=headers)
            response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)

            response_json = response.json()
        
        if "choices" in response_json and response_json["choices"]:
            return response_json["choices"][0]["message"]["content"].strip()
        
        return "⚠️ Error: No valid response received from the AI."

    except requests.exceptions.Timeout:
        return "⚠️ Error: API request timed out. Please try again later."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error: API request failed. {str(e)}"









# import requests
# import os
# import streamlit as st
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# @st.cache_data(ttl=300, show_spinner=False)  # ✅ Hides loading message
# def fetch_groq_response(prompt):
#     """Fetch response from Groq API and cache it."""
#     if not GROQ_API_KEY:
#         return "Error: API key not found. Please set the GROQ_API_KEY environment variable."

#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
    
#     data = {
#         "model": "llama-3.3-70b-versatile",
#         "messages": [{"role": "user", "content": prompt}],
#         "max_tokens": 200
#     }

#     try:
#         with st.spinner():  # ✅ Keeps spinner but hides extra text
#             response = requests.post(GROQ_ENDPOINT, json=data, headers=headers)
#             response_json = response.json()
        
#         if "choices" in response_json and response_json["choices"]:
#             return response_json["choices"][0]["message"]["content"]
#         return "Error: No valid response received from API."
    
#     except requests.exceptions.RequestException:
#         return "Error: API request failed. Please check your internet connection and API key."
