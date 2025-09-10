#!/usr/bin/env python3
"""
Page de Recherche - Fantasy RAG Streamlit
=========================================

Page modulaire pour la recherche sémantique d'assets 3D.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le package fantasy_rag au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def show():
    """Affiche la page de recherche"""
    
    st.header("🔍 Recherche d'Assets 3D")
    
    # Utiliser l'ancienne interface de recherche
    try:
        # Import de l'ancienne interface
        from streamlit_app import display_main_search
        
        # Paramètres de langue
        language = st.session_state.get("global_language", "français")
        
        # Afficher l'interface de recherche
        display_main_search(language)
        
    except ImportError as e:
        st.error(f"Erreur import interface recherche: {e}")
        
        # Interface de recherche basique en fallback
        st.subheader("Recherche Basique")
        
        query = st.text_input("Rechercher un asset 3D:", placeholder="Ex: épée magique")
        
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Langue", ["français", "english"])
        with col2:
            max_results = st.slider("Résultats max", 1, 20, 5)
        
        if st.button("🔍 Rechercher") and query:
            with st.spinner("Recherche en cours..."):
                st.info("Interface de recherche en cours de migration...")
                st.write(f"Recherche: '{query}' en {language}")
                st.write(f"Maximum {max_results} résultats")

if __name__ == "__main__":
    show()
