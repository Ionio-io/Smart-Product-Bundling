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

# Paths for embeddings storage
BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
CSV_EMBEDDINGS = BASE_PATH + "csv_embeddings.npy"
LLM_EMBEDDINGS = BASE_PATH + "llm_embeddings.npy"
BIGBASKET_CSV = BASE_PATH + "BigBasket Products.csv"

# Load Sentence Transformer model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Ionio-ai/retail_embedding_classifier_v1").to(device)

def load_json(file_path):
    """Loads JSON file and returns its content. Returns an empty dictionary if the file doesn't exist."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found. Returning empty dictionary.")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"⚠️ Error: Failed to decode {file_path}. Returning empty dictionary.")
        return {}

def generate_product_bundle(product_name):
    """Passes product name through LLM to get a list of 5 complementary products."""

    system_prompt = """
    Suggest product bundles to the user
    
    <THINK>
    - Think about the product's daily use and where it is usually used
    - Then think what other products is the product used with
    - Think about the possible products which are complementary to its usage
    - Think about more and more things which the product can be paired with, maintain variation and diversity, think as much as you want
    -If the {product_name} is tea leaves for example, the consumer is thinking of making tea. In that case suggest products that can be used to make tea.
    Another example can be when the {product_name} is body wash. In that case, the consumer is thinking of buying toiletries. Think of products that can be bought with body wash like, hand wash or face wash.
    Think of the most common daily use things one can make with {product_name} where applicable or what are the most common daily use items {product_name} can be paired with.
    </THINK>
    Suggest product bundles to the user.

    Output in **valid JSON format** like this:

    {
        "bundles": [
            {"Product": "Product Name | Short description"},
            {"Product": "Product Name | Short description"},
            {"Product": "Product Name | Short description"},
            {"Product": "Product Name | Short description"},
            {"Product": "Product Name | Short description"}
        ]
    }

    - Only return **valid JSON**.
    - Do not include any explanations, markdown formatting, or additional text.
    """

    user_prompt = f"""
     A customer is purchasing a {product_name}. Suggest the **top 5 most relevant complementary products** that are often bought together in a bundle.
    Think of the context in which the consumer is buying the product. If the {product_name} is tea leaves for example, the consumer is thinking of making tea. In that case suggest products that can be used to make tea.
    Another example can be when the {product_name} is body wash. In that case, the consumer is thinking of buying toiletries. Think of products that can be bought with body wash like, hand wash or face wash.
    Think of the most common daily use things one can make with {product_name} where applicable or what are the most common daily use items {product_name} can be paired with.

    ### **Guidelines for Selecting Products:**
    1. **Directly Useful**: The products should serve an **immediate and practical** purpose when used together.
    2. **Enhance Daily Routine**: Select items that **improve convenience, efficiency, or experience** in everyday use.
    3. **Popular Pairings**: Choose products that **are commonly bought together in supermarkets, stores, or online**.
    4. **Avoid Redundancy**: Ensure **no two products serve the exact same function** in the bundle.

    ---

    ### **Examples of Well-Designed Daily-Use Product Bundles:**

    ✅ **Main Product: Hand Wash**
       - **Moisturizer** - Keeps hands soft and prevents dryness after frequent washing.
       - **Hand Sanitizer** - A portable option for maintaining hygiene on the go.
       - **Paper Towels** - Convenient for drying hands quickly.
       - **Liquid Soap Refill** - Ensures continued use without running out.
       - **Nail Brush** - Helps in deep cleaning and removing dirt under nails.

    ✅ **Main Product: Toothpaste**
       - **Toothbrush Set** - Ensures proper dental hygiene with fresh bristles.
       - **Mouthwash** - Provides extra protection against bad breath and bacteria.
       - **Dental Floss** - Helps clean between teeth where the brush cannot reach.
       - **Tongue Cleaner** - Improves oral hygiene by removing bacteria from the tongue.
       - **Toothpaste Squeezer** - Helps extract every last bit from the tube.

    ✅ **Main Product: Dishwashing Liquid**
       - **Sponge Scrubber** - Essential for scrubbing off grease and stains.
       - **Microfiber Kitchen Towel** - Quick-drying towel for wiping dishes.
       - **Dish Rack** - Helps air-dry plates and utensils after washing.
       - **Gloves** - Protects hands from detergent and prolonged water exposure.
       - **Garbage Bags** - For easy disposal of food waste after washing dishes.

    ✅ **Main Product: Shampoo**
       - **Conditioner** - Complements the shampoo for smooth and manageable hair.
       - **Hair Serum** - Adds shine and reduces frizz after washing.
       - **Hair Towel Wrap** - Absorbs excess water quickly and reduces drying time.
       - **Scalp Massager** - Helps distribute shampoo and improves circulation.
       - **Dry Shampoo** - A quick alternative for freshening up hair between washes.

    Product: {product_name}

    Suggest exactly **5 highly relevant complementary products** based on **real-world use cases**.

    Think of the product's daily use and the **most commonly bought together items**.
    Ensure the output **matches the JSON format**.
    """

    for _ in range(3):  # Retry up to 3 times in case of API failure
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                temperature=0.5,
                presence_penalty=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}  # ✅ Correct format
            )

            # ✅ Properly decode JSON string into a dictionary
            bundle_products = json.loads(response.choices[0].message.content)

            return bundle_products["bundles"]  # ✅ Extract JSON correctly

        except (json.JSONDecodeError, openai.APIError, KeyError, TypeError) as e:
            print(f"⚠️ LLM JSON decoding error: {e}")

    raise ValueError("⚠️ LLM did not return valid JSON!")


def generate_embeddings(llm_products):
    """Generates embeddings only for LLM products and saves them."""

    if not os.path.exists(CSV_EMBEDDINGS):
        raise FileNotFoundError("⚠️ CSV embeddings file not found! Ensure csv_embeddings.npy exists.")

    llm_texts = [item["Product"] for item in llm_products]
    llm_embeddings = model.encode(llm_texts, convert_to_numpy=True, device=device)

    np.save(LLM_EMBEDDINGS, llm_embeddings)


def compute_similarity():
    """Loads precomputed CSV embeddings and newly generated LLM embeddings, then computes cosine similarity."""
    
    csv_embeddings = np.load(CSV_EMBEDDINGS)
    llm_embeddings = np.load(LLM_EMBEDDINGS)

    return cosine_similarity(llm_embeddings, csv_embeddings)


def get_top_5_similar_products(llm_products, similarity_matrix):
    """Retrieve top 5 similar products for each LLM-generated product, using indices from CSV."""

    df_products = pd.read_csv(BIGBASKET_CSV)

    results = []
    for i, llm_product in enumerate(llm_products):
        top_5_indices = np.argsort(similarity_matrix[i])[-5:][::-1]

        similar_products = []
        for idx in top_5_indices:
            if idx < len(df_products):
                similar_products.append({
                    "Product": f"{df_products.iloc[idx]['product']} | {df_products.iloc[idx]['description']}"
                })

        results.append({
            "LLM Product": llm_product["Product"],
            "Similar Products": similar_products
        })

    return results

def generate_llm_bundle(product_name, similar_products):
    """Passes the top similar products through GPT-4 to generate exactly 3 product bundles."""

    formatted_products = json.dumps(similar_products, indent=4)

    system_prompt = """
    Suggest product bundles to the user
    <THINK>
    - Think about the product's daily use and where it is usually used
    - Then think what other products is the product used with
    - Think about the possible products which are complementary to its usage
    - Think about more and more things which the product can be paired with, maintain variation and diversity, think as much as you want
    -If the {product_name} is tea leaves for example, the consumer is thinking of making tea. In that case suggest products that can be used to make tea.
    Another example can be when the {product_name} is body wash. In that case, the consumer is thinking of buying toiletries. Think of products that can be bought with body wash like, hand wash or face wash.
    Think of the most common daily use things one can make with {product_name} where applicable or what are the most common daily use items {product_name} can be paired with.
    </THINK>
    Generate **exactly 3 product bundles** in JSON format.

    Output in this format:

    {
        "bundles": [
            {
                "Bundle Name": "bundle_name",
                "Bundle Description": "short description",
                "Products": [
                    {"Product": "Product Name | Short description"},
                    {"Product": "Product Name | Short description"},
                    {"Product": "Product Name | Short description"},
                    {"Product": "Product Name | Short description"},
                    {"Product": "Product Name | Short description"}
                ],
                "Marketing Copy": "marketing copy for the bundle"
            }
        ]
    }

    - Return **only JSON output**.
    - Ensure the output is **valid JSON**.
    """

    user_prompt = f"""
    You are an expert in product bundling. Your task is to generate exactly **3 unique bundles** using the provided products.

    **Main Product:** {product_name}  
    **Available Products for Bundling:**  
    {formatted_products}

    **Guidelines:**  
    - Each bundle must contain **exactly 5 products**, including the main product.  
    - Provide a **bundle number, bundle name, bundle description, and marketing copy.**  
    - Follow the format strictly, without explanations.
    """

    for _ in range(3):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                temperature=0.5,
                presence_penalty=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}  # ✅ Correct format
            )

            # ✅ Properly decode JSON string into a dictionary
            bundle_data = json.loads(response.choices[0].message.content)

            return bundle_data["bundles"]  # ✅ Extract JSON correctly

        except (json.JSONDecodeError, openai.APIError, KeyError, TypeError) as e:
            print(f"⚠️ LLM JSON decoding error: {e}")

    raise ValueError("⚠️ LLM did not return valid JSON!")


