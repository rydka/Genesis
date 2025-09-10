#!/usr/bin/env python3
"""
Fantasy RAG - Application Streamlit Principale
==============================================

Point d'entrée principal pour l'interface Streamlit modulaire.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le package fantasy_rag au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Point d'entrée principal"""

    st.set_page_config(
        page_title="Fantasy RAG",
        page_icon="🏰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Navigation
    st.sidebar.title("Fantasy RAG")
    
    # Import des pages
    try:
        from .pages import search, generation, analytics, admin

        pages = {
            "Recherche": search.show,
            "Generation": generation.show,
            "Analytics": analytics.show,
            "Admin": admin.show
        }

        # Sélection de page
        page = st.sidebar.selectbox("Navigation", list(pages.keys()))

        # Affichage de la page
        pages[page]()

    except ImportError as e:
        st.error(f"Erreur import pages: {e}")
        st.info("Utilisation du mode de compatibilité...")

        # Fallback vers l'ancienne interface
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from streamlit_app import main as old_main
            old_main()
        except ImportError:
            st.error("Interface non disponible")

if __name__ == "__main__":
    main()
