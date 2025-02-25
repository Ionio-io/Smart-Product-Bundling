import json
import sys
from utils import (
    generate_product_bundle, generate_embeddings,
    compute_similarity, get_top_5_similar_products, generate_llm_bundle
)

# Paths for storing data
BASE_PATH = "C:/Users/jaysa/OneDrive/Desktop/Ionio/"
FINAL_BUNDLE_JSON = BASE_PATH + "final_bundle.json"

def save_json(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def generate_complementary_products(product_name):
    """Generates a list of 5 complementary products using the first LLM call."""
    print("\n🔍 Generating complementary products...")
    bundle_products = generate_product_bundle(product_name)
    print("✅ Complementary products generated.")
    return bundle_products

def find_similar_products(bundle_products):
    """Finds the top 5 similar products for each LLM-generated product."""
    print("\n📊 Computing product similarities...")
    generate_embeddings(bundle_products)
    similarity_matrix = compute_similarity()
    
    similar_products_data = get_top_5_similar_products(bundle_products, similarity_matrix)
    print("✅ Similar products identified.")

    # Flatten similar products for the next step
    similar_products = []
    for bundle in similar_products_data:
        similar_products.extend(bundle["Similar Products"])
    
    return similar_products

def create_final_product_bundle(product_name, similar_products):
    """Generates the final product bundle and saves it."""
    print("\n📦 Generating final product bundles...")
    final_bundles = generate_llm_bundle(product_name, similar_products)

    final_bundle_output = {
        "Product Name": product_name,
        "Similar Products": similar_products,
        "Bundles": final_bundles
    }

    # ✅ Save to JSON before returning
    save_json(final_bundle_output, FINAL_BUNDLE_JSON)
    print(f"✅ Final product bundles saved successfully to {FINAL_BUNDLE_JSON}!")

    return final_bundle_output  # Return the final JSON to confirm it exists

def main(product_name):
    """Main pipeline to generate product bundles."""
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
