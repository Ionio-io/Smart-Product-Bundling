import json
import sys
from utils import (
    save_json, generate_product_bundle, generate_embeddings,
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
    save_json(bundle_products, BUNDLE_JSON)  # ✅ Save first LLM output

    # Step 2: Compute Similarity & Save Similar Products
    generate_embeddings(bundle_products)
    similarity_matrix = compute_similarity()
    similar_products_data = get_top_5_similar_products(bundle_products, similarity_matrix)
    save_json(similar_products_data, SIMILAR_PRODUCTS_JSON)  # ✅ Save top 5 matches per product

    # Step 3: Prepare Similar Products for the Second LLM Call
    similar_products = []
    for bundle in similar_products_data:
        similar_products.extend(bundle["Similar Products"])  # ✅ Flatten the structure

    # Step 4: Generate LLM Bundles with Updated Products
    final_bundles = generate_llm_bundle(product_name, similar_products)  # ✅ Second LLM call

    # ✅ Ensure all required data is saved
    final_bundle_output = {
        "Product Name": product_name,
        "Similar Products": similar_products,  # ✅ From similar_products.json
        "Bundles": final_bundles  # ✅ Output from the second LLM call
    }

    save_json(final_bundle_output, FINAL_BUNDLE_JSON)
    print("\n✅ Final bundles successfully saved to JSON!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Error: Please provide a product name.")
    else:
        main(sys.argv[1])