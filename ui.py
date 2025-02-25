import streamlit as st
import subprocess
import json
import time
import os

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

# Page Layout
st.set_page_config(layout="wide")

st.title("🔹 AI-Powered Product Bundle Generator")

# Input for product name
product_name = st.text_input("Enter Product Name:", value="", key="product_input")

if st.button("Generate Bundle"):
    with st.spinner("Generating bundles... Please wait."):
        subprocess.run(["python", BASE_PATH + "main.py", product_name], check=True)

    # Wait for the JSON file to be available
    bundle_data = load_json(FINAL_BUNDLE_JSON)

    if bundle_data:
        st.markdown(f"## Generated Bundles for: {product_name}")

        if "Bundles" in bundle_data and isinstance(bundle_data["Bundles"], list):
            bundles = bundle_data["Bundles"]  # ✅ This is now a list of dictionaries

            # Layout with equal width columns for 3 bundles
            cols = st.columns(3)

            for i, bundle in enumerate(bundles[:3]):  # Display only first 3 bundles
                if isinstance(bundle, dict):  # ✅ Ensure bundle is a dictionary
                    bundle_name = bundle.get("Bundle Name", "Unnamed Bundle")
                    bundle_desc = bundle.get("Bundle Description", "No description available.")
                    marketing_copy = bundle.get("Marketing Copy", "No marketing copy provided.")
                    
                    # Extract product list correctly
                    products_list = [item["Product"] for item in bundle.get("Products", [])]

                    with cols[i]:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #1e1e1e;
                                padding: 20px;
                                border-radius: 10px;
                                box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.1);
                                min-height: 450px;
                                color: white;
                                text-align: center;
                            ">
                                <h3 style="color: white;">🎁 {bundle_name}</h3>
                                <p style="font-style: italic; color: white;">{bundle_desc}</p>
                                <hr style="border: 1px solid #444;">
                                <h4 style="text-align: left; color: white;">📦 Included Products:</h4>
                                <ul style="text-align: left; color: white;">
                                    {"".join([f"<li>{product}</li>" for product in products_list])}
                                </ul>
                                <hr style="border: 1px solid #444;">
                                <p style="font-weight: bold; color: white;">📢 {marketing_copy}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    print(f"⚠️ Warning: Unexpected data format for bundle {i+1}")
    else:
        st.error("⚠️ Error: Failed to load generated bundles. Please try again.")
