#!/usr/bin/env python3
"""
Page Analytics - Fantasy RAG Streamlit
======================================

Page modulaire pour les analytics et statistiques du système.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le package fantasy_rag au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def show():
    """Affiche la page analytics"""
    
    st.header("📊 Analytics & Statistiques")
    
    # Utiliser l'ancienne interface d'analytics
    try:
        # Import de l'interface avancée
        from streamlit_advanced import display_analytics_dashboard
        
        # Afficher l'interface d'analytics
        display_analytics_dashboard()
        
    except ImportError as e:
        st.error(f"Erreur import interface analytics: {e}")
        
        # Interface analytics basique en fallback
        st.subheader("Statistiques Système")
        
        # Métriques de base
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Assets Totaux", "10+", "2")
        with col2:
            st.metric("Descriptions", "10+", "2")
        with col3:
            st.metric("Rendus", "60+", "12")
        with col4:
            st.metric("Recherches", "50+", "5")
        
        # Graphiques
        st.subheader("Utilisation")
        st.info("Graphiques d'analytics en cours de migration...")
        
        # Logs récents
        st.subheader("Activité Récente")
        st.text_area("Logs système", "Interface analytics en cours de migration...", height=200)

if __name__ == "__main__":
    show()
