import streamlit as st
import subprocess
import json
import time
import os
import re

# Paths
BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"

def load_json(file_path, retries=5, wait_time=2):
    """Loads JSON file with retries in case it's not ready."""
    for attempt in range(retries):
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: JSON file not fully written. Retrying... ({attempt+1}/{retries})")
        time.sleep(wait_time)
    
    print("⚠️ Error: Unable to load JSON file after retries.")
    return None

def clean_product_name(product):
    """Removes the leading number and dot (e.g., '1. ') but keeps the full product name."""
    return re.sub(r"^\d+\.\s*", "", product)  # Removes only the leading number and dot

# Page Layout
st.set_page_config(layout="wide", page_title="AI-Powered Product Bundle Generator")

# Set light theme styles
st.markdown(
    """
    <style>
        body {
            background-color: #f8f9fa;
            color: black;
        }
        .stApp {
            background-color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: black !important;
        }
        .bundle-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            min-height: 450px;
            color: black;
            text-align: center;
        }
        .bundle-container h3 {
            color: #d9534f;
        }
        .bundle-container p {
            color: black;
        }
        .bundle-container ul {
            text-align: left;
            color: black;
        }
        .bundle-container hr {
            border: 1px solid #ccc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# **Title**
st.markdown("<h1 style='text-align: center; color: black;'>🔹 AI-Powered Product Bundle Generator</h1>", unsafe_allow_html=True)

# Input for product name
product_name = st.text_input("Enter Product Name:", value="", key="product_input")

if st.button("Generate Bundle"):
    with st.spinner("Generating bundles... Please wait."):
        subprocess.run(["python", BASE_PATH + "main.py", product_name], check=True)

    # Wait for the JSON file to be available
    bundle_data = load_json(FINAL_BUNDLE_JSON)

    if bundle_data:
        # **Subheading for generated bundles**
        st.markdown(f"<h2 style='text-align: center; color: black;'>Generated Bundles for: {product_name}</h2>", unsafe_allow_html=True)

        if "Bundles" in bundle_data and isinstance(bundle_data["Bundles"], list):
            bundles = bundle_data["Bundles"]  # ✅ This is now a list of dictionaries

            # Layout with equal width columns for 3 bundles
            cols = st.columns(3)

            for i, bundle in enumerate(bundles[:3]):  # Display only first 3 bundles
                if isinstance(bundle, dict):  # ✅ Ensure bundle is a dictionary
                    bundle_name = bundle.get("Bundle Name", "Unnamed Bundle")
                    bundle_desc = bundle.get("Bundle Description", "No description available.")
                    marketing_copy = bundle.get("Marketing Copy", "No marketing copy provided.")
                    
                    # Extract product list correctly and clean the product names
                    products_list = [
                        clean_product_name(item["Product"])  # ✅ Removes only the number, keeps the name intact
                        for item in bundle.get("Products", [])
                    ]

                    with cols[i]:
                        st.markdown(
                            f"""
                            <div class="bundle-container">
                                <h3>🎁 {bundle_name}</h3>
                                <p style="font-style: italic;">{bundle_desc}</p>
                                <hr>
                                <h4>📦 Included Products:</h4>
                                <ul>
                                    {"".join([f"<li>{product}</li>" for product in products_list])}
                                </ul>
                                <hr>
                                <p style="font-weight: bold;">📢 {marketing_copy}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    print(f"⚠️ Warning: Unexpected data format for bundle {i+1}")
    else:
        st.error("⚠️ Error: Failed to load generated bundles. Please try again.")