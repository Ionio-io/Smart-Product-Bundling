import json
import pandas as pd
import sys
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
    
    print(f"\n✅ Running pipeline for product: {product_name}")

    # Step 1: First LLM Pass - Generate Complementary Products
    bundle_products = generate_product_bundle(product_name)
    save_json(bundle_products, BUNDLE_JSON)

    # Step 2: Compute Similarity & Save Similar Products
    generate_embeddings(bundle_products)
    similarity_matrix = compute_similarity()
    similar_products = get_top_5_similar_products(bundle_products, similarity_matrix)
    save_json(similar_products, SIMILAR_PRODUCTS_JSON)

    # Step 3: Second LLM Pass - Generate Bundle Name, Description, and Marketing Copy
    final_bundle_output = {
        "Product Name": product_name,  # ✅ Always use the original product name
        "Similar Products": [],
        "Bundle Details": {}
    }

    for bundle in similar_products:
        top_similar = bundle["Similar Products"]

        if top_similar:
            final_bundle = generate_llm_bundle(product_name, top_similar)  # ✅ Pass original product_name
            final_bundle_output["Similar Products"] = top_similar
            final_bundle_output["Bundle Details"] = final_bundle
        else:
            print(f"⚠️ No similar products found for {product_name}.")

    # Save final bundle
    save_json(final_bundle_output, FINAL_BUNDLE_JSON)
    print("\n✅ Final bundle successfully generated!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Error: Please provide a product name.")
    else:
        main(sys.argv[1])
