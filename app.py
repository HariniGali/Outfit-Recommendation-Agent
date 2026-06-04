import traceback

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
from flask_cors import CORS

import config
from agent import FashionStylistAgent

app = Flask(__name__)
CORS(app)

# Initialise agent and catalog once at startup
agent = FashionStylistAgent(
    gemini_api_key=config.GEMINI_API_KEY,
    image_directory=config.IMAGE_DIR,
    chroma_persist_directory=config.CHROMA_DB_PATH,
)

_catalog_df: pd.DataFrame | None = None


def get_catalog() -> pd.DataFrame:
    global _catalog_df
    if _catalog_df is None:
        _catalog_df = agent.load_catalog(config.CATALOG_CSV_PATH)
    return _catalog_df


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/images/<path:filename>')
def serve_image(filename):
    if config.USE_S3:
        return redirect(f'{config.IMAGE_BASE_URL}{filename}')
    return send_from_directory(config.IMAGE_DIR, filename)


@app.route('/api/products')
def get_products():
    try:
        df = get_catalog()
        sample = df.sample(n=min(20, len(df)))
        products = [
            {
                'product_id': str(row.get('product_id', '')),
                'name': str(row.get('name', '')),
                'description': str(row.get('description', '')),
                'category': str(row.get('terms', '')),
                'image_filename': str(row.get('image_downloads', '')),
            }
            for _, row in sample.iterrows()
        ]
        return jsonify(products)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        product = data['product']
        occasion = data.get('occasion', 'casual everyday')
        style_vibe = data.get('style_vibe', 'modern minimalist')

        outfit = agent.complete_outfit(
            anchor_product_name=product['name'],
            anchor_product_description=product.get('description', ''),
            image_filename=product.get('image_filename'),
            occasion=occasion,
            style_vibe=style_vibe,
            top_k=3,
        )
        return jsonify(outfit)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Loading catalog from {config.CATALOG_CSV_PATH} ...')
    catalog = get_catalog()
    print(f'Ingesting {len(catalog)} products into ChromaDB ...')
    agent.ingest_catalog(catalog)
    print(f'\nStarting server at http://{config.FLASK_HOST}:{config.FLASK_PORT}')
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)
