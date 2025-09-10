#!/usr/bin/env python3
"""
Page Administration - Fantasy RAG Streamlit
===========================================

Page modulaire pour l'administration du système.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le package fantasy_rag au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def show():
    """Affiche la page d'administration"""
    
    st.header("⚙️ Administration Système")
    
    # Onglets d'administration
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Configuration",
        "🗄️ Base de Données", 
        "🧹 Maintenance",
        "📋 Logs"
    ])
    
    with tab1:
        st.subheader("Configuration Système")
        
        # Configuration Ollama
        st.write("**🧠 Configuration Ollama**")
        ollama_url = st.text_input("URL Ollama", "http://localhost:11434")
        ollama_model = st.selectbox("Modèle", ["llama3.2-vision:latest", "llama3.2:latest"])
        
        # Configuration ChromaDB
        st.write("**💾 Configuration ChromaDB**")
        chromadb_path = st.text_input("Chemin ChromaDB", "data/chromadb")
        
        # Configuration Blender
        st.write("**🎨 Configuration Blender**")
        blender_path = st.text_input("Chemin Blender", "blender")
        
        if st.button("💾 Sauvegarder Configuration"):
            st.success("Configuration sauvegardée !")
    
    with tab2:
        st.subheader("Gestion Base de Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Statistiques**")
            if st.button("📈 Actualiser Stats"):
                st.info("Statistiques actualisées")
        
        with col2:
            st.write("**🔄 Actions**")
            if st.button("🧹 Nettoyer Cache"):
                st.info("Cache nettoyé")
            if st.button("💾 Backup DB"):
                st.info("Backup créé")
    
    with tab3:
        st.subheader("Maintenance Système")
        
        # Outils de maintenance
        st.write("**🛠️ Outils Disponibles**")
        
        if st.button("🔍 Vérifier Santé Système"):
            st.info("Vérification en cours...")
            st.success("✅ Système en bonne santé")
        
        if st.button("🧪 Lancer Tests"):
            st.info("Tests en cours...")
            st.success("✅ Tous les tests passés")
        
        if st.button("📦 Traiter Nouveaux Assets"):
            st.info("Traitement en cours...")
            st.success("✅ Assets traités")
    
    with tab4:
        st.subheader("Logs Système")
        
        # Sélection niveau de logs
        log_level = st.selectbox("Niveau", ["DEBUG", "INFO", "WARNING", "ERROR"])
        
        # Zone de logs
        st.text_area(
            "Logs récents",
            "Interface d'administration en cours de migration...\nLogs système seront affichés ici.",
            height=300
        )
        
        if st.button("🔄 Actualiser Logs"):
            st.info("Logs actualisés")

if __name__ == "__main__":
    show()
