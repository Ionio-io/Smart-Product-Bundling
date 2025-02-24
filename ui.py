import streamlit as st
import subprocess
from utils import load_json

# Paths
BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"

# Page Layout
st.set_page_config(layout="wide")

st.title("🔹 AI-Powered Product Bundle Generator")

# Input for product name
product_name = st.text_input("Enter Product Name:", value="", key="product_input")

if st.button("Generate Bundle"):
    with st.spinner("Generating bundles... Please wait."):
        subprocess.run(["python", BASE_PATH + "main.py", product_name], check=True)

    # Load final bundle
    bundle_data = load_json(FINAL_BUNDLE_JSON)

    # Ensure the correct product name is shown in UI
    st.markdown(f"## Generated Bundles for: {product_name}")

    # Display bundles in a structured format
    if "Bundles" in bundle_data:
        bundles = bundle_data["Bundles"].split("---")  # Splitting bundles

        # Layout with equal width columns for 3 bundles
        cols = st.columns(3)

        for i, bundle in enumerate(bundles[:3]):  # Display only first 3 bundles
            # Extract only one instance of each part (Fix duplication)
            try:
                bundle_name = bundle.split('Bundle Name: ')[1].split('\n')[0].strip()
                bundle_desc = bundle.split('Bundle Description: ')[1].split('\n')[0].strip()
                marketing_copy = bundle.split('Marketing Copy: ')[1].strip()

                # Extract product list correctly
                product_lines = bundle.split("Bundle:")[1].split("Marketing Copy:")[0].strip().split("\n")
                products_list = [line.strip() for line in product_lines if line.strip()]

            except IndexError:
                continue  # Skip if parsing fails

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