import streamlit as st
import subprocess
from utils import load_json

BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"

st.title("🔹 Product Bundle Generator")
product_name = st.text_input("Enter Product Name:")

if st.button("Generate Bundle"):
    subprocess.run(["python", BASE_PATH + "main.py", product_name], check=True)

    # Load final bundle and display it
    bundle_data = load_json(FINAL_BUNDLE_JSON)

    # ✅ Ensure the correct product name is shown in UI
    bundle_data["Product Name"] = product_name

    # Display the formatted JSON in Streamlit
    st.json(bundle_data, expanded=True)
