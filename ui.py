import streamlit as st
import subprocess
from utils import load_json

BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"

st.title("🔹 Product Bundle Generator")
product_name = st.text_input("Enter Product Name:")

if st.button("Generate Bundle"):
    subprocess.run(["python", BASE_PATH + "main.py", product_name])
    bundle_data = load_json(FINAL_BUNDLE_JSON)
    st.write(bundle_data)
