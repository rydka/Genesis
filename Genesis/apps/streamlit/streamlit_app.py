#!/usr/bin/env python3
"""
Fantasy RAG - Interface Streamlit Moderne
==========================================

Interface utilisateur avancée pour la recherche et exploration d'assets 3D fantastiques
avec support multilingue, visualisation interactive et génération assistée.
"""

import streamlit as st
import requests
import json
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import base64
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="Fantasy RAG - Assets 3D Fantastiques",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'API
API_BASE_URL = "http://localhost:5000"

def load_css():
    """Charge les styles CSS personnalisés"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .asset-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .asset-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    .search-stats {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .view-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 1rem 0;
    }
    
    .view-image {
        border-radius: 8px;
        border: 2px solid #eee;
        transition: border-color 0.3s ease;
    }
    
    .view-image:hover {
        border-color: #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def check_api_connection():
    """Vérifie la connexion à l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=5)
        return response.status_code == 200
    except:
        return False

def search_assets(query: str, language: str = "français", max_results: int = 10) -> Dict[str, Any]:
    """Effectue une recherche d'assets"""
    try:
        payload = {
            "query": query,
            "language": language,
            "max_results": max_results,
            "min_similarity": -1.0
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/search",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_asset_details(asset_id: str) -> Dict[str, Any]:
    """Récupère les détails d'un asset"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/asset/{asset_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_api_stats() -> Dict[str, Any]:
    """Récupère les statistiques de l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_asset_card(asset: Dict[str, Any], show_details: bool = False):
    """Affiche une carte d'asset"""
    
    with st.container():
        st.markdown('<div class="asset-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Image principale (front view)
            if asset.get("renders"):
                front_render = next((r for r in asset["renders"] if r["view_name"] == "front"), asset["renders"][0])
                image_url = f"{API_BASE_URL}/api/image/{asset['asset_id']}/front.png"
                
                try:
                    st.image(image_url, caption=f"Vue principale", use_column_width=True)
                except:
                    st.error("❌ Image non disponible")
        
        with col2:
            # Informations de l'asset
            st.subheader(f"🏰 {asset.get('asset_name', 'Asset inconnu')}")
            
            # Métadonnées
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("📊 Similarité", f"{asset.get('similarity_score', 0):.3f}")
                st.metric("📝 Mots", asset.get('word_count', 'N/A'))
            
            with col2b:
                st.metric("🖼️ Vues", asset.get('render_count', 0))
                st.metric("🏷️ Statut", asset.get('asset_status', 'Inconnu'))
            
            # Description (tronquée)
            description = asset.get('description', 'Aucune description disponible')
            if len(description) > 200:
                description = description[:200] + "..."
            
            st.write("**Description :**")
            st.write(description)
            
            # Boutons d'action
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button(f"🔍 Voir détails", key=f"details_{asset['asset_id']}"):
                    st.session_state.selected_asset = asset['asset_id']
                    st.rerun()
            
            with col_btn2:
                if st.button(f"🖼️ Toutes les vues", key=f"views_{asset['asset_id']}"):
                    st.session_state.show_views = asset['asset_id']
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_asset_views(asset_id: str):
    """Affiche toutes les vues d'un asset"""
    
    st.subheader(f"🖼️ Toutes les vues - {asset_id}")
    
    views = ["front", "back", "left", "right", "top", "iso_3d"]
    view_names = {
        "front": "Vue de face",
        "back": "Vue de dos", 
        "left": "Vue de gauche",
        "right": "Vue de droite",
        "top": "Vue du dessus",
        "iso_3d": "Vue isométrique"
    }
    
    # Grille 2x3 pour les vues
    for row in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            view_idx = row * 3 + col_idx
            if view_idx < len(views):
                view = views[view_idx]
                with cols[col_idx]:
                    image_url = f"{API_BASE_URL}/api/image/{asset_id}/{view}.png"
                    try:
                        st.image(image_url, caption=view_names[view], use_column_width=True)
                    except:
                        st.error(f"❌ {view_names[view]} non disponible")
    
    if st.button("🔙 Retour à la recherche"):
        if 'show_views' in st.session_state:
            del st.session_state.show_views
        st.rerun()

def main():
    """Interface principale"""

    # Chargement des styles
    load_css()

    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🏰 Fantasy RAG - Assets 3D Fantastiques</h1>
        <p>Recherche sémantique multilingue et exploration d'assets 3D</p>
    </div>
    """, unsafe_allow_html=True)

    # Vérification de la connexion API
    if not check_api_connection():
        st.error("❌ **Connexion à l'API impossible**")
        st.info("💡 Assurez-vous que le serveur Flask est démarré : `python -m fantasy_rag.web.app`")
        st.stop()

    # Navigation par onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Recherche",
        "🔄 Comparaison",
        "🤖 Génération",
        "📊 Analytics",
        "🎛️ Recherche Avancée"
    ])

    # Sidebar pour les paramètres globaux
    with st.sidebar:
        st.header("⚙️ Paramètres")

        # Langue globale
        language = st.selectbox(
            "🌐 Langue",
            ["français", "english"],
            index=0,
            key="global_language"
        )

        # Statistiques
        st.header("📈 Statistiques")
        stats = get_api_stats()
        if stats.get("success"):
            st.metric("📦 Assets totaux", stats.get("total_assets", 0))
            st.metric("📝 Descriptions", stats.get("total_descriptions", 0))
            st.metric("🖼️ Rendus", stats.get("total_renders", 0))

        st.header("ℹ️ À propos")
        st.info("""
        **Fantasy RAG** utilise :
        - 🧠 Llama3.2-Vision pour les descriptions
        - 🔍 Recherche sémantique multilingue
        - 🎨 Rendus Blender 6 vues
        - 💾 ChromaDB pour le stockage
        - 🎯 MeshFormer pour la génération
        """)

    # Gestion des états de session pour les vues
    if 'show_views' in st.session_state:
        display_asset_views(st.session_state.show_views)
        return

    # Contenu des onglets
    with tab1:
        display_main_search(language)

    with tab2:
        # Import des modules avancés
        try:
            from streamlit_advanced import display_asset_comparison
            display_asset_comparison()
        except ImportError:
            st.error("Module de comparaison non disponible")

    with tab3:
        try:
            from streamlit_advanced import display_generation_assistant
            display_generation_assistant()
        except ImportError:
            st.error("Module de génération non disponible")

    with tab4:
        try:
            from streamlit_advanced import display_analytics_dashboard
            display_analytics_dashboard()
        except ImportError:
            st.error("Module d'analytics non disponible")

    with tab5:
        try:
            from streamlit_advanced import display_advanced_search
            display_advanced_search()
        except ImportError:
            st.error("Module de recherche avancée non disponible")

def display_main_search(language: str):
    """Affiche l'interface de recherche principale"""

    # Paramètres de recherche
    col_params1, col_params2 = st.columns(2)

    with col_params1:
        max_results = st.slider(
            "📊 Nombre de résultats",
            min_value=1,
            max_value=20,
            value=5,
            key="main_search_results"
        )

    with col_params2:
        search_mode = st.selectbox(
            "🎯 Mode de recherche",
            ["Sémantique", "Exact", "Hybride"],
            key="main_search_mode"
        )
    
    # Interface de recherche principale
    st.header("🔍 Recherche d'Assets")

    # Barre de recherche
    col_search, col_btn = st.columns([4, 1])

    with col_search:
        query = st.text_input(
            "Rechercher des assets 3D...",
            placeholder="Ex: épée magique, dragon, armure de chevalier...",
            key="main_search_query"
        )

    with col_btn:
        st.write("")  # Espacement
        search_clicked = st.button("🔍 Rechercher", type="primary", key="main_search_btn")

    # Exemples de recherche
    st.write("**💡 Exemples :** `épée`, `personnage armé`, `warrior`, `magical sword`")

    # Recherche automatique ou sur clic
    if query and (search_clicked or query != st.session_state.get('last_main_query', '')):
        st.session_state.last_main_query = query

        with st.spinner(f"🔍 Recherche en cours pour '{query}'..."):
            results = search_assets(query, language, max_results)

        if results.get("success"):
            # Statistiques de recherche
            st.markdown(f"""
            <div class="search-stats">
                <strong>📊 Résultats :</strong> {results.get('total_found', 0)} assets trouvés
                en {results.get('processing_time', 0):.2f}s
                <br><strong>🎯 Mode :</strong> {search_mode}
            </div>
            """, unsafe_allow_html=True)

            # Options d'affichage
            col_view1, col_view2 = st.columns(2)
            with col_view1:
                view_mode = st.selectbox(
                    "👁️ Mode d'affichage",
                    ["Cartes détaillées", "Liste compacte", "Grille d'images"],
                    key="view_mode"
                )

            with col_view2:
                sort_by = st.selectbox(
                    "📊 Trier par",
                    ["Similarité", "Nom", "Nombre de mots", "Nombre de vues"],
                    key="sort_by"
                )

            # Affichage des résultats
            if results.get('results'):
                st.header(f"🎯 Résultats de recherche ({len(results['results'])})")

                # Tri des résultats
                sorted_results = results['results'].copy()
                if sort_by == "Nom":
                    sorted_results.sort(key=lambda x: x.get('asset_name', ''))
                elif sort_by == "Nombre de mots":
                    sorted_results.sort(key=lambda x: x.get('word_count', 0), reverse=True)
                elif sort_by == "Nombre de vues":
                    sorted_results.sort(key=lambda x: x.get('render_count', 0), reverse=True)
                # Similarité est déjà triée par défaut

                # Affichage selon le mode choisi
                if view_mode == "Cartes détaillées":
                    for i, asset in enumerate(sorted_results):
                        display_asset_card(asset)
                        if i < len(sorted_results) - 1:
                            st.divider()

                elif view_mode == "Liste compacte":
                    for asset in sorted_results:
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            st.write(f"**{asset.get('asset_name', 'Asset')}**")
                        with col2:
                            st.write(f"{asset.get('similarity_score', 0):.3f}")
                        with col3:
                            st.write(f"{asset.get('word_count', 'N/A')}")
                        with col4:
                            if st.button("👁️", key=f"view_{asset['asset_id']}"):
                                st.session_state.selected_asset = asset['asset_id']
                                st.rerun()

                elif view_mode == "Grille d'images":
                    # Affichage en grille 3 colonnes
                    for i in range(0, len(sorted_results), 3):
                        cols = st.columns(3)
                        for j, col in enumerate(cols):
                            if i + j < len(sorted_results):
                                asset = sorted_results[i + j]
                                with col:
                                    image_url = f"{API_BASE_URL}/api/image/{asset['asset_id']}/front.png"
                                    st.image(image_url, caption=asset.get('asset_name', 'Asset'), use_column_width=True)
                                    st.write(f"Sim: {asset.get('similarity_score', 0):.3f}")
            else:
                st.warning("🤷 Aucun résultat trouvé pour cette recherche.")
                st.info("💡 Essayez avec des termes différents ou en anglais.")

        else:
            st.error(f"❌ **Erreur de recherche :** {results.get('error', 'Erreur inconnue')}")

    elif not query:
        # Page d'accueil sans recherche
        st.info("👆 Entrez un terme de recherche pour explorer les assets 3D fantastiques !")

        # Affichage des statistiques générales
        stats = get_api_stats()
        if stats.get("success"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📦 {stats.get('total_assets', 0)}</h3>
                    <p>Assets 3D</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 {stats.get('total_descriptions', 0)}</h3>
                    <p>Descriptions</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🖼️ {stats.get('total_renders', 0)}</h3>
                    <p>Rendus</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
