import streamlit as st
import requests

st.set_page_config(page_title="Restaurant Rating Predictor")

st.title("🍽️ Restaurant Rating Predictor")

st.markdown("Enter restaurant details below:")

# Input fields
name = st.text_input("Restaurant Name")
location = st.text_input("Location")
cuisine = st.text_input("Cuisine Type")
price_range = st.selectbox("Price Range", ["$", "$$", "$$$", "$$$$"])
review_text = st.text_area("Review Text")  # Added for API compatibility

if st.button("Predict Rating"):
    if not all([name, location, cuisine, price_range, review_text]):
        st.warning("⚠️ Please fill in all the fields.")
    else:
        data = {
            "name": name,
            "location": location,
            "categories": cuisine,   # Changed key to 'categories'
            "price_range": price_range,
            "text": review_text      # Added review text
        }

        try:
            response = requests.post("http://localhost:8005/predict", json=data)
            if response.status_code == 200:
                prediction = response.json().get("predicted_rating")
                st.success(f"⭐ Predicted Rating: {prediction}")
            else:
                st.error(f"❌ Error from API: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the API. Is it running on port 8005?")
