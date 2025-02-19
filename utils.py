import json
import numpy as np
import torch
import openai
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths for data storage
BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
BUNDLE_JSON = BASE_PATH + "bundle.json"
SIMILAR_PRODUCTS_JSON = BASE_PATH + "similar_products.json"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"
CSV_EMBEDDINGS = BASE_PATH + "csv_embeddings.npy"
LLM_EMBEDDINGS = BASE_PATH + "llm_embeddings.npy"
BIGBASKET_CSV = BASE_PATH + "BigBasket Products.csv"

# Load Sentence Transformer model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Ionio-ai/retail_embedding_classifier_v1").to(device)

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def save_json(data, file_path):
    """Save data to JSON file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def generate_product_bundle(product_name):
    """Passes product name through LLM to get a list of 5 complementary products."""
    prompt = f"""
    A customer is purchasing a {product_name}. Suggest the **top 5 most relevant complementary products** that are often bought together in a bundle.
    The products should be **directly useful** with the main product and **enhance the user experience**.

    **Format**: Provide results in the format **Product Name - Short description.** 

    Now, provide **5 relevant bundled products** for **{product_name}** in the same format.
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    bundle_items = response.choices[0].message.content.strip().split("\n")
    product_list = []
    for item in bundle_items:
        if " - " in item:
            name, desc = item.split(" - ", 1)
            product_list.append({"Product": name.strip(), "Description": desc.strip()})

    return product_list

def generate_embeddings(llm_products):
    """🔹 Generates embeddings only for LLM products and saves them. Loads CSV embeddings from a precomputed file."""
    
    # Load existing CSV embeddings (do not recompute)
    if not os.path.exists(CSV_EMBEDDINGS):
        raise FileNotFoundError("⚠️ CSV embeddings file not found! Ensure csv_embeddings.npy exists.")

    # Generate LLM embeddings
    llm_texts = [f"{item['Product']} - {item['Description']}" for item in llm_products]
    llm_embeddings = model.encode(llm_texts, convert_to_numpy=True, device=device)

    # Save LLM embeddings
    np.save(LLM_EMBEDDINGS, llm_embeddings)

def compute_similarity():
    """🔹 Loads precomputed CSV embeddings and newly generated LLM embeddings, then computes cosine similarity."""
    
    # Load existing embeddings
    csv_embeddings = np.load(CSV_EMBEDDINGS)  # ✅ Precomputed CSV embeddings
    llm_embeddings = np.load(LLM_EMBEDDINGS)  # ✅ Just generated LLM embeddings

    return cosine_similarity(llm_embeddings, csv_embeddings)

def get_top_5_similar_products(llm_products, similarity_matrix):
    """Retrieve top 5 similar products for each LLM-generated product, using indices from CSV."""
    
    # Load BigBasket CSV file
    df_products = pd.read_csv(BIGBASKET_CSV)

    # Ensure column names are correct
    if "product" not in df_products.columns or "description" not in df_products.columns:
        raise ValueError("CSV file missing required columns: 'product' and 'description'")

    results = []
    for i, llm_product in enumerate(llm_products):
        # Get top 5 most similar product indices and their similarity scores
        top_5_indices = np.argsort(similarity_matrix[i])[-5:][::-1]  
        top_5_scores = similarity_matrix[i][top_5_indices]  # Get similarity scores

        similar_products = []
        for idx, score in zip(top_5_indices, top_5_scores):
            if idx < len(df_products):  # Ensure index is within bounds
                similar_products.append({
                    "Product": df_products.iloc[idx]["product"],  # ✅ Get from CSV
                    "Description": df_products.iloc[idx]["description"],  # ✅ Get from CSV
                    "Similarity Score": float(score)  # ✅ Store similarity score
                })

        print(f"✅ {i+1}. **{llm_product['Product']} -> {len(similar_products)} Similar Products Found**")

        results.append({
            "LLM Product": llm_product["Product"],
            "LLM Description": llm_product["Description"],
            "Similar Products": similar_products
        })

    return results


def generate_llm_bundle(product_name, similar_products):
    """Passes the top 5 similar products through GPT-4 to generate a marketing copy."""
    
    # Ensure similar products exist
    if not similar_products:
        return "**⚠️ No similar products found for this bundle.**"

    similar_product_list = "\n".join([f"{item['Product']} - {item['Description']} (Score: {item['Similarity Score']:.4f})" for item in similar_products])

    prompt = f"""
    You are an expert in marketing and product bundling. Generate a bundle name, description, and marketing copy.

    **Main Product:** {product_name}
    **Available Products:**
    {similar_product_list}

    **Example Format:**
    **Bundle Name:** [Creative Name]
    **Bundle Description:** [Short compelling description]

    **Marketing Copy:**  
    "This exclusive bundle ensures you have everything needed for an amazing experience. Buy now!"
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
