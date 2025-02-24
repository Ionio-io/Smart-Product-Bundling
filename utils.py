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
    Think of the context in which the consumer is buying the product. If the {product_name} is tea leaves for example, the consumer is thinking of making tea. In that case suggest products that can be used to make tea.
    Another example can be when the {product_name} is body wash. In that case, the consumer is thinking of buying toiletries. Think of products that can be bought with body wash like, hand wash or face wash.
    Think of the most common daily use things one can make with {product_name} where applicalble or what are the most common daily use items {product_name} can be paired with.
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
    The products should be **directly useful** with the main product and **enhance the user experience**.

    **Format**: Provide results in the format **Product Name - Short description.** 

    Now, provide **exactly 5 relevant bundled products** for **{product_name}** in the same format.
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
    """Generates embeddings only for LLM products and saves them. Loads CSV embeddings from a precomputed file."""
    
    # Load existing CSV embeddings
    if not os.path.exists(CSV_EMBEDDINGS):
        raise FileNotFoundError("⚠️ CSV embeddings file not found! Ensure csv_embeddings.npy exists.")

    # Generate LLM embeddings
    llm_texts = [f"{item['Product']} - {item['Description']}" for item in llm_products]
    llm_embeddings = model.encode(llm_texts, convert_to_numpy=True, device=device)

    # Save LLM embeddings
    np.save(LLM_EMBEDDINGS, llm_embeddings)

def compute_similarity():
    """Loads precomputed CSV embeddings and newly generated LLM embeddings, then computes cosine similarity."""
    
    # Load embeddings
    csv_embeddings = np.load(CSV_EMBEDDINGS)
    llm_embeddings = np.load(LLM_EMBEDDINGS)

    return cosine_similarity(llm_embeddings, csv_embeddings)

def get_top_5_similar_products(llm_products, similarity_matrix):
    """Retrieve top 5 similar products for each LLM-generated product, using indices from CSV."""

    # Load BigBasket CSV file
    df_products = pd.read_csv(BIGBASKET_CSV)

    results = []
    for i, llm_product in enumerate(llm_products):
        top_5_indices = np.argsort(similarity_matrix[i])[-5:][::-1]

        similar_products = []
        for idx in top_5_indices:
            if idx < len(df_products):
                similar_products.append({
                    "Product": df_products.iloc[idx]["product"],
                    "Description": df_products.iloc[idx]["description"]
                })

        print(f"✅ {i+1}. **{llm_product['Product']} -> {len(similar_products)} Similar Products Found**")

        results.append({
            "LLM Product": llm_product["Product"],
            "LLM Description": llm_product["Description"],
            "Similar Products": similar_products
        })

    return results

def generate_llm_bundle(product_name, similar_products):
    """Passes the top similar products through GPT-4 to generate exactly 3 product bundles."""

    formatted_products = "\n".join([f"{item['Product']} - {item['Description']}" for item in similar_products])

    prompt = f"""
You are an expert in product bundling. Your task is to generate exactly **3 unique bundles** using the provided products.
Think of the context in which the consumer is buying the product. If the {product_name} is tea leaves for example, the consumer is thinking of making tea. In that case suggest products that can be used to make tea.
    Another example can be when the {product_name} is body wash. In that case, the consumer is thinking of buying toiletries. Think of products that can be bought with body wash like, hand wash or face wash.
    ### **Guidelines for Selecting Products:**
1. **Directly Useful**: The products should serve an **immediate and practical** purpose when used together.
2. **Enhance Daily Routine**: Select items that **improve convenience, efficiency, or experience** in everyday use.
3. **Popular Pairings**: Choose products that **are commonly bought together in supermarkets, stores, or online**.
4. **Avoid Redundancy**: Ensure **no two products serve the exact same function** in the bundle.
5. **High Variety**: Each bundle you suggest should have a different use case.

**Main Product:** {product_name}  
**Available Products for Bundling:**  
{formatted_products}

**Guidelines:**  
- Each bundle must contain **exactly 5 products**, including the main product.  
- Ensure all selected products **complement each other**.  
- Provide a **bundle number, bundle name, bundle description, and marketing copy.**  
- Follow the format strictly, without explanations.

**Expected Format:**

Bundle 1:  
Bundle Name: [Creative Name]  
Bundle Description: [Short compelling description]  

Bundle:  
1. {product_name} - [Product Description]  
2. [Product 2 - Short Description]  
3. [Product 3 - Short Description]  
4. [Product 4 - Short Description]  
5. [Product 5 - Short Description]  

Marketing Copy:  
[Engaging marketing copy encouraging purchase]

---

Now, generate **exactly 3 bundles** in this format.
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()