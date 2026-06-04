import os
import json
import logging
from typing import List, Dict, Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FashionStylistAgent:
    """AI Fashion Stylist backed by Gemini + ChromaDB vector search."""

    def __init__(
        self,
        gemini_api_key: str,
        image_directory: str,
        chroma_persist_directory: str = './chroma_fashion_db',
        embedding_model: str = 'all-MiniLM-L6-v2',
    ):
        self.image_directory = image_directory
        self.chroma_persist_directory = chroma_persist_directory

        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

        self.client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name='fashion_products',
            embedding_function=self.embedding_function,
            metadata={'description': 'Fashion product catalog'},
        )

        logger.info('Fashion Stylist Agent initialized')
        logger.info(f'Image directory: {self.image_directory}')

    # ------------------------------------------------------------------
    # Catalog helpers
    # ------------------------------------------------------------------

    def get_image_path(self, image_filename: str) -> Optional[str]:
        if not image_filename or pd.isna(image_filename) or image_filename == '':
            return None
        full_path = os.path.join(self.image_directory, image_filename)
        return full_path if os.path.exists(full_path) else None

    def load_catalog(self, csv_path: str) -> pd.DataFrame:
        logger.info(f'Loading catalog from {csv_path}')
        df = pd.read_csv(csv_path)

        if 'product_id' not in df.columns:
            df['product_id'] = df.index.astype(str)

        if 'image_downloads' in df.columns:
            df['image_full_path'] = df['image_downloads'].apply(self.get_image_path)
            images_found = df['image_full_path'].notna().sum()
            logger.info(f'Found {images_found} images out of {len(df)} products')

        logger.info(f'Loaded {len(df)} products')
        return df

    def _make_product_doc(self, row: pd.Series) -> str:
        parts = []
        for field, label in [('name', 'Name'), ('description', 'Description'),
                              ('florence2_description', 'Visual'), ('terms', 'Category')]:
            if field in row and pd.notna(row[field]):
                parts.append(f'{label}: {row[field]}')
        return ' | '.join(parts)

    def ingest_catalog(self, df: pd.DataFrame):
        logger.info('Starting catalog ingestion into ChromaDB')

        try:
            self.client.delete_collection(name='fashion_products')
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name='fashion_products',
            embedding_function=self.embedding_function,
            metadata={'description': 'Fashion product catalog'},
        )

        documents, metadatas, ids = [], [], []

        for idx, row in df.iterrows():
            documents.append(self._make_product_doc(row))
            metadata = {
                'product_id': str(row.get('product_id', idx)),
                'name': str(row.get('name', '')),
                'category': str(row.get('terms', '')),
            }
            if 'image_downloads' in row and pd.notna(row['image_downloads']):
                metadata['image_filename'] = str(row['image_downloads'])
            metadatas.append(metadata)
            ids.append(f'product_{idx}')

        batch_size = 500
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )
            logger.info(f'Ingested batch {i // batch_size + 1}: {i}–{end}')

        logger.info(f'Successfully ingested {len(documents)} products')

    # ------------------------------------------------------------------
    # Recommendation logic
    # ------------------------------------------------------------------

    def _load_image(self, image_filename: str) -> Optional[Image.Image]:
        full_path = self.get_image_path(image_filename)
        if not full_path:
            return None
        try:
            img = Image.open(full_path)
            return img.convert('RGB') if img.mode != 'RGB' else img
        except Exception as e:
            logger.warning(f'Could not load image {image_filename}: {e}')
            return None

    def get_outfit_recommendations(
        self,
        product_name: str,
        product_description: str,
        image_filename: Optional[str] = None,
        occasion: str = 'casual everyday',
        style_vibe: str = 'modern minimalist',
    ) -> Dict:
        prompt = f"""You are an expert fashion stylist. Given a fashion item, suggest complementary pieces to create a complete, stylish outfit.

ANCHOR ITEM:
Name: {product_name}
Description: {product_description}

STYLING CONTEXT:
Occasion: {occasion}
Style Vibe: {style_vibe}

Please suggest 3-4 complementary items that would complete this outfit. For each item, provide:
1. Item category only including - tshirts, sweatshirts, trousers, jeans, shoes, jackets
2. Detailed description of style, color, material, and fit
3. Why it complements the anchor item

Format your response as a JSON array with this structure:
[
  {{
    "category": "trousers",
    "description": "slim-fit black cotton trousers with a tapered leg",
    "reasoning": "creates a sleek silhouette that balances the volume of the jacket"
  }},
  ...
]

Be specific about colors, materials, cuts, and styles."""

        content = [prompt]
        image_used = False

        if image_filename:
            img = self._load_image(image_filename)
            if img:
                content = [img, prompt]
                image_used = True

        response = self.gemini_model.generate_content(content)
        text = response.text

        if '```json' in text:
            json_str = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            json_str = text.split('```')[1].split('```')[0].strip()
        else:
            json_str = text.strip()

        recommendations = json.loads(json_str)
        return {
            'anchor_product': product_name,
            'occasion': occasion,
            'style_vibe': style_vibe,
            'recommendations': recommendations,
            'image_used': image_used,
        }

    def search_similar_products(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        products = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            products.append({
                'product_id': meta['product_id'],
                'name': meta['name'],
                'category': meta['category'],
                'similarity_score': 1 - results['distances'][0][i],
                'image_filename': meta.get('image_filename', ''),
                'image_path': self.get_image_path(meta.get('image_filename', '')),
            })
        return products

    def complete_outfit(
        self,
        anchor_product_name: str,
        anchor_product_description: str,
        image_filename: Optional[str] = None,
        occasion: str = 'casual everyday',
        style_vibe: str = 'modern minimalist',
        top_k: int = 3,
    ) -> Dict:
        gemini_response = self.get_outfit_recommendations(
            anchor_product_name,
            anchor_product_description,
            image_filename,
            occasion,
            style_vibe,
        )

        outfit_items = []
        for rec in gemini_response['recommendations']:
            query = f"{rec['category']} {rec['description']}"
            matches = self.search_similar_products(query=query, n_results=top_k)
            outfit_items.append({'recommendation': rec, 'matched_products': matches})

        return {
            'anchor_product': {
                'name': anchor_product_name,
                'description': anchor_product_description,
                'image_filename': image_filename,
                'image_path': self.get_image_path(image_filename) if image_filename else None,
            },
            'context': {'occasion': occasion, 'style_vibe': style_vibe},
            'image_analysis_used': gemini_response.get('image_used', False),
            'outfit_items': outfit_items,
        }
