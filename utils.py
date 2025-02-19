import json
import numpy as np
import torch
import openai
import os
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

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    bundle_items = response["choices"][0]["message"]["content"].strip().split("\n")
    product_list = []
    for item in bundle_items:
        if " - " in item:
            name, desc = item.split(" - ", 1)
            product_list.append({"Product": name.strip(), "Description": desc.strip()})

    return product_list

def generate_embeddings(csv_products, llm_products):
    """Generate embeddings and save them to NumPy files."""
    csv_texts = [f"{row['product']} - {row['description']}" for _, row in csv_products.iterrows()]
    llm_texts = [f"{item['Product']} - {item['Description']}" for item in llm_products]

    csv_embeddings = model.encode(csv_texts, convert_to_numpy=True, device=device)
    llm_embeddings = model.encode(llm_texts, convert_to_numpy=True, device=device)

    np.save(CSV_EMBEDDINGS, csv_embeddings)
    np.save(LLM_EMBEDDINGS, llm_embeddings)

def compute_similarity():
    """Load embeddings and compute cosine similarity."""
    csv_embeddings = np.load(CSV_EMBEDDINGS)
    llm_embeddings = np.load(LLM_EMBEDDINGS)
    return cosine_similarity(llm_embeddings, csv_embeddings)

def get_top_5_similar_products(llm_products, df_filtered, similarity_matrix):
    """Retrieve top 5 similar products for each product in LLM products."""
    results = []
    for i, llm_product in enumerate(llm_products):
        top_5_indices = np.argsort(similarity_matrix[i])[-5:][::-1]
        similar_products = [
            {"Product": df_filtered.iloc[idx]["product"], "Description": df_filtered.iloc[idx]["description"]}
            for idx in top_5_indices
        ]
        results.append({
            "LLM Product": llm_product["Product"],
            "LLM Description": llm_product["Description"],
            "Similar Products": similar_products
        })
    return results

def generate_llm_bundle(product_name, similar_products):
    """Passes the top 5 similar products through GPT-4 to generate a marketing copy."""
    similar_product_list = "\n".join([f"{item['Product']} - {item['Description']}" for item in similar_products])

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

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"].strip()
