import json
import pandas as pd
from utils import (
    load_json, save_json, generate_product_bundle, generate_embeddings,
    compute_similarity, get_top_5_similar_products, generate_llm_bundle
)

# Paths
BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
BUNDLE_JSON = BASE_PATH + "bundle.json"
SIMILAR_PRODUCTS_JSON = BASE_PATH + "similar_products.json"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"

def main(product_name):
    """Handles the full pipeline: LLM pass, similarity, and final bundle generation."""
    
    # Step 1: First LLM Pass - Generate Complementary Products
    bundle_products = generate_product_bundle(product_name)
    save_json(bundle_products, BUNDLE_JSON)

    # Step 2: Compute Similarity & Save Similar Products
    df_filtered = pd.read_csv(BASE_PATH + "products.csv")
    generate_embeddings(df_filtered, bundle_products)
    similarity_matrix = compute_similarity()
    similar_products = get_top_5_similar_products(bundle_products, df_filtered, similarity_matrix)
    save_json(similar_products, SIMILAR_PRODUCTS_JSON)

    # Step 3: Second LLM Pass - Generate Bundle Name, Description, and Marketing Copy
    final_bundle = generate_llm_bundle(product_name, similar_products[0]["Similar Products"])
    save_json({"Product Name": product_name, "Bundle Details": final_bundle}, FINAL_BUNDLE_JSON)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
