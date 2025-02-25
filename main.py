import json
import sys
from utils import (
    save_json, generate_product_bundle, generate_embeddings,
    compute_similarity, get_top_5_similar_products, generate_llm_bundle
)

# Paths for storing data
BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
BUNDLE_JSON = BASE_PATH + "bundle.json"
SIMILAR_PRODUCTS_JSON = BASE_PATH + "similar_products.json"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"


def generate_complementary_products(product_name):
    """
    Generates a list of 5 complementary products using the first LLM call.
    Saves the output to BUNDLE_JSON.
    """
    print("\n🔍 Generating complementary products...")
    bundle_products = generate_product_bundle(product_name)
    save_json(bundle_products, BUNDLE_JSON)
    print("✅ Complementary products saved.")
    return bundle_products


def find_similar_products(bundle_products):
    """
    Computes similarity between generated products and database products.
    Retrieves the top 5 similar products for each LLM-generated product.
    Saves the results to SIMILAR_PRODUCTS_JSON.
    """
    print("\n📊 Computing product similarities...")
    generate_embeddings(bundle_products)
    similarity_matrix = compute_similarity()
    
    similar_products_data = get_top_5_similar_products(bundle_products, similarity_matrix)
    save_json(similar_products_data, SIMILAR_PRODUCTS_JSON)
    print("✅ Similar products saved.")

    # Flattening the similar products into a list for the next step
    similar_products = []
    for bundle in similar_products_data:
        similar_products.extend(bundle["Similar Products"])
    
    return similar_products


def create_final_product_bundle(product_name, similar_products):
    """
    Uses the second LLM call to generate the final product bundle.
    Saves the final result to FINAL_BUNDLE_JSON.
    """
    print("\n📦 Generating final product bundles...")
    final_bundles = generate_llm_bundle(product_name, similar_products)

    final_bundle_output = {
        "Product Name": product_name,
        "Similar Products": similar_products,
        "Bundles": final_bundles
    }

    save_json(final_bundle_output, FINAL_BUNDLE_JSON)
    print("✅ Final product bundles saved successfully!")


def main(product_name):
    """
    Main pipeline that orchestrates:
    1. Generating complementary products (First LLM call)
    2. Finding similar products using embeddings
    3. Generating the final product bundle (Second LLM call)
    """
    print(f"\n🚀 Running product bundling pipeline for: {product_name}")

    # Step 1: Generate complementary products
    bundle_products = generate_complementary_products(product_name)

    # Step 2: Find similar products
    similar_products = find_similar_products(bundle_products)

    # Step 3: Generate final product bundle
    create_final_product_bundle(product_name, similar_products)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Error: Please provide a product name.")
    else:
        main(sys.argv[1])
