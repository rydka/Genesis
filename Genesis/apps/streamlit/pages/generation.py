#!/usr/bin/env python3
"""
Page de Génération - Fantasy RAG Streamlit
==========================================

Page modulaire pour la génération de mesh 3D avec MeshFormer.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le package fantasy_rag au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def show():
    """Affiche la page de génération"""
    
    st.header("🎨 Génération de Mesh 3D")
    
    # Utiliser l'ancienne interface de génération
    try:
        # Import de l'interface avancée
        from streamlit_advanced import display_generation_assistant
        
        # Afficher l'interface de génération
        display_generation_assistant()
        
    except ImportError as e:
        st.error(f"Erreur import interface génération: {e}")
        
        # Interface de génération basique en fallback
        st.subheader("Génération Basique")
        
        prompt = st.text_area(
            "Prompt de génération:", 
            placeholder="Décrivez l'asset 3D à générer (ex: épée elfique élégante)"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            asset_type = st.selectbox("Type", ["weapon", "armor", "creature", "object"])
        with col2:
            style = st.selectbox("Style", ["fantastique", "elfique", "nain", "magique", "sombre"])
        with col3:
            complexity = st.slider("Complexité", 1, 10, 5)
        
        if st.button("🎨 Générer Mesh") and prompt:
            with st.spinner("Génération en cours..."):
                st.info("Interface de génération en cours de migration...")
                st.write(f"Prompt: '{prompt}'")
                st.write(f"Type: {asset_type}, Style: {style}, Complexité: {complexity}")

if __name__ == "__main__":
    show()
