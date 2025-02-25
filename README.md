
# 🛍️ AI-Powered Product Bundle Generator

An intelligent product bundling tool that leverages Large Language Models (LLMs) to dynamically generate complementary product bundles based on a given main product. This tool ensures that businesses can optimize product recommendations, enhance customer experience, and increase sales by offering the most relevant product combinations.


## 👀 Sneak Peek
🚀 Automatically generate product bundles based on the main product 

🎯 Identify top similar products using AI-driven embeddings & similarity analysis

📊 Create highly optimized bundles using LLM-generated marketing strategies

⚡ Build dynamic UI with Streamlit for an interactive experience
## 🤔 How It Works?
1️⃣ Get the Main Product Input

Users enter the main product (e.g., "Flax Seeds"), and the system begins analyzing potential complementary products.

2️⃣ Generate Complementary Products (LLM Call #1)

The tool queries GPT-4o to generate a list of 5 complementary products that are frequently bought together with the main product.

3️⃣ Find Similar Products using Embeddings

The system generates text embeddings for the LLM-suggested products.
It compares them with a product database to identify top 5 most similar products using cosine similarity.

4️⃣ Generate Final Bundles (LLM Call #2)

A second LLM call takes the top similar products and formulates 3 highly optimized bundles.

Each bundle includes:

✅ Bundle Name

✅ Bundle Description

✅ 5 carefully selected products

✅ Marketing Copy to boost sales

5️⃣ Display Bundles in the Streamlit UI

The generated bundles are presented in a visually appealing and easy-to-navigate dashboard, ensuring a seamless user experience.

## 🚀 Getting Started

🛠 Prerequisites

Python 3.10+

OpenAI API Key

Streamlit

SentenceTransformers for embeddings

💻 Installation

1️⃣ Clone the Repository

```bash
git clone https://github.com/Ionio-io/Smart-Product-Bundling.git

cd Smart-Product-Bundling
```
2️⃣ Install Dependencies

```bash
pip install openai streamlit pandas numpy sentence-transformers scikit-learn python-dotenv
```

3️⃣ Set Up API Key

Create a .env file and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key
```

4️⃣ Run the Streamlit App
```bash
streamlit run ui.py
```

## 📂 Project Structure

```bash
📦 Smart-Product-Bundling
│-- 📜 main.py         # Main pipeline for product bundling
│-- 📜 utils.py        # LLM interactions, embeddings, similarity calculations
│-- 📜 ui.py           # Streamlit UI for bundle visualization
│-- 📜 .env            # Stores OpenAI API Key
│-- 📜 requirements.txt # Dependencies list (if available)
```