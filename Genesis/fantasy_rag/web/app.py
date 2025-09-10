#!/usr/bin/env python3
"""
Fantasy RAG Web Interface

Interactive web interface for searching and exploring fantasy assets
using the complete Fantasy RAG system.
"""

from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import logging
import time
from typing import Dict, List, Any, Optional
import base64
import mimetypes

# Fantasy RAG imports
from ..database.chromadb_manager import ChromaDBManager, DatabaseConfig
from ..embeddings.multilingual_search import MultilingualSearchEngine, MultilingualSearchQuery
from ..description.prompt_templates import DescriptionLanguage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'fantasy_rag_secret_key_2024'

# Global components
db_manager = None
search_engine = None

def initialize_components():
    """Initialize Fantasy RAG components"""
    global db_manager, search_engine
    
    try:
        # Initialize database manager
        config = DatabaseConfig(
            db_path="ressources/data/fantasy_rag_db"
        )
        
        db_manager = ChromaDBManager(config)
        if not db_manager.initialize():
            logger.error("Failed to initialize database manager")
            return False
        
        # Initialize search engine
        search_engine = MultilingualSearchEngine(
            db_path="ressources/data/fantasy_rag_db",
            collection_name="asset_descriptions"
        )
        
        if not search_engine.initialize():
            logger.error("Failed to initialize search engine")
            return False
        
        logger.info("✅ Fantasy RAG components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        return False

@app.route('/')
def index():
    """Main search interface"""
    
    # Get database statistics
    stats = {}
    if db_manager:
        stats = db_manager.get_database_stats()
    
    return render_template('index.html', stats=stats)

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for semantic search"""
    
    try:
        data = request.get_json()
        
        query_text = data.get('query', '').strip()
        language = data.get('language', 'français')
        max_results = min(int(data.get('max_results', 10)), 20)
        min_similarity = float(data.get('min_similarity', -1.0))  # Accept negative similarities
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': 'Query text is required'
            })
        
        if not search_engine:
            return jsonify({
                'success': False,
                'error': 'Search engine not initialized'
            })
        
        # Convert language string to enum
        try:
            lang_enum = DescriptionLanguage.FRENCH if language.lower() in ['français', 'french', 'fr'] else DescriptionLanguage.ENGLISH
        except:
            lang_enum = DescriptionLanguage.FRENCH
        
        # Create search query
        search_query = MultilingualSearchQuery(
            text=query_text,
            source_language=lang_enum,
            max_results=max_results,
            min_similarity=min_similarity
        )
        
        # Perform search
        start_time = time.time()
        search_result = search_engine.search_multilingual(search_query)
        search_time = time.time() - start_time
        
        # Format results
        results = []
        for result in search_result.results:
            # Get asset details
            asset = db_manager.get_asset(result.asset_id) if db_manager else None
            renders = db_manager.get_asset_renders(result.asset_id) if db_manager else []
            
            result_data = {
                'asset_id': result.asset_id,
                'similarity_score': round(result.similarity_score, 3),
                'description': result.description,
                'language': result.metadata.get('language', 'unknown'),
                'word_count': len(result.description.split()),
                'asset_name': asset.original_filename if asset else 'Unknown',
                'asset_category': asset.category.value if asset else 'unknown',
                'asset_status': asset.status.value if asset else 'unknown',
                'render_count': len(renders),
                'renders': [
                    {
                        'view_name': render.view_name,
                        'image_path': str(render.image_path),
                        'resolution': f"{render.resolution[0]}x{render.resolution[1]}" if isinstance(render.resolution, tuple) else render.resolution,
                        'file_size': render.file_size
                    }
                    for render in renders[:6]  # Limit to 6 views
                ]
            }
            results.append(result_data)
        
        return jsonify({
            'success': True,
            'query': query_text,
            'language': language,
            'total_found': search_result.total_found,
            'processing_time': round(search_time, 3),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/stats')
def api_stats():
    """API endpoint for database statistics"""
    
    try:
        if not db_manager:
            return jsonify({
                'success': False,
                'error': 'Database manager not initialized'
            })
        
        stats = db_manager.get_database_stats()
        
        # Add search engine stats
        if search_engine:
            search_stats = search_engine.get_language_statistics()
            stats['search_engine'] = search_stats
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/asset/<asset_id>')
def api_asset_details(asset_id):
    """API endpoint for detailed asset information"""
    
    try:
        if not db_manager:
            return jsonify({
                'success': False,
                'error': 'Database manager not initialized'
            })
        
        # Get asset details
        asset = db_manager.get_asset(asset_id)
        if not asset:
            return jsonify({
                'success': False,
                'error': 'Asset not found'
            })
        
        # Get renders
        renders = db_manager.get_asset_renders(asset_id)
        
        # Get descriptions
        descriptions = db_manager.get_asset_descriptions(asset_id)
        
        asset_data = {
            'asset_id': asset.asset_id,
            'original_filename': asset.original_filename,
            'file_path': str(asset.file_path),
            'file_hash': asset.file_hash,
            'category': asset.category.value,
            'status': asset.status.value,
            'current_stage': asset.current_stage.value,
            'file_size': asset.file_size,
            'file_format': asset.file_format,
            'created_at': asset.created_at,
            'updated_at': asset.updated_at,
            'processing_attempts': asset.processing_attempts,
            'last_error': asset.last_error,
            'processing_time': asset.processing_time,
            'tags': asset.tags,
            'custom_metadata': asset.custom_metadata,
            'renders': [
                {
                    'render_id': render.render_id,
                    'view_name': render.view_name,
                    'image_path': str(render.image_path),
                    'resolution': f"{render.resolution[0]}x{render.resolution[1]}" if isinstance(render.resolution, tuple) else render.resolution,
                    'samples': render.samples,
                    'render_engine': render.render_engine,
                    'render_time': render.render_time,
                    'file_size': render.file_size,
                    'created_at': render.created_at,
                    'has_transparency': render.has_transparency,
                    'dominant_colors': render.dominant_colors
                }
                for render in renders
            ],
            'descriptions': [
                {
                    'description_id': desc.description_id,
                    'language': desc.language,
                    'description_text': desc.description_text,
                    'model_name': desc.model_name,
                    'generation_time': desc.generation_time,
                    'word_count': desc.word_count,
                    'target_length': desc.target_length,
                    'quality_score': desc.quality_score,
                    'created_at': desc.created_at,
                    'views_analyzed': desc.views_analyzed
                }
                for desc in descriptions
            ]
        }
        
        return jsonify({
            'success': True,
            'asset': asset_data
        })
        
    except Exception as e:
        logger.error(f"Asset details error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/image/<asset_id>/<view_name>.png')
def api_image(asset_id, view_name):
    """Serve rendered images for specific asset and view"""

    try:
        # Construct the actual filename: asset_id_view_name.png
        filename = f"{asset_id}_{view_name}.png"

        # Get the project root directory (go up from fantasy_rag/web/)
        project_root = Path(__file__).parent.parent.parent
        full_path = project_root / "ressources" / "data" / "renders" / asset_id / filename

        logger.debug(f"Looking for image: {full_path}")

        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}")
            return jsonify({
                'success': False,
                'error': f'Image not found: {filename}'
            }), 404

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if not mime_type:
            mime_type = 'image/png'

        logger.debug(f"Serving image: {full_path}")
        return send_file(str(full_path), mimetype=mime_type)

    except Exception as e:
        logger.error(f"Image serving error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/image/<path:image_path>')
def api_image_fallback(image_path):
    """Fallback endpoint for direct image paths"""

    try:
        # Get the project root directory (go up from fantasy_rag/web/)
        project_root = Path(__file__).parent.parent.parent
        full_path = project_root / "ressources" / "data" / "renders" / image_path

        logger.debug(f"Fallback looking for image: {full_path}")

        if not full_path.exists():
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if not mime_type:
            mime_type = 'image/png'

        return send_file(str(full_path), mimetype=mime_type)

    except Exception as e:
        logger.error(f"Image serving error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/asset/<asset_id>')
def asset_detail(asset_id):
    """Asset detail page"""
    return render_template('asset_detail.html', asset_id=asset_id)

@app.route('/gallery')
def gallery():
    """Asset gallery page"""
    return render_template('gallery.html')

if __name__ == '__main__':
    print("🚀 Starting Fantasy RAG Web Interface")
    print("=" * 50)
    
    if initialize_components():
        print("✅ Components initialized successfully")
        print("🌐 Starting web server...")
        print("   URL: http://localhost:5000")
        print("   Press Ctrl+C to stop")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize components")
        print("   Make sure the Fantasy RAG database exists")
        exit(1)
