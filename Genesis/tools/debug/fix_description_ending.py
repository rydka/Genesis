#!/usr/bin/env python3
"""
Fix the description ending to have proper punctuation
"""

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_description_ending():
    """Fix the description ending"""
    
    print("🔧 Fixing Description Ending")
    print("=" * 40)
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        from fantasy_rag.database.schema_definitions import DescriptionRecord
        
        # Initialize database
        config = DatabaseConfig(db_path="ressources/data/fantasy_rag_db")
        db_manager = ChromaDBManager(config)
        
        if not db_manager.initialize():
            print("❌ Failed to initialize database")
            return False
        
        # Get current description
        desc_collection = db_manager.collections[db_manager.config.descriptions_collection]
        all_docs = desc_collection.get()
        
        if not all_docs['ids']:
            print("❌ No descriptions found")
            return False
        
        # Get the first description
        doc_id = all_docs['ids'][0]
        metadata = all_docs['metadatas'][0]
        current_text = all_docs['documents'][0]
        
        print(f"📝 Current description ends with: '{current_text[-50:]}'")
        
        # Fix the ending - remove the extra words and add proper punctuation
        # The description should end with "...prêt pour l'aventure et le combat."
        
        # Find the natural ending point
        target_ending = "prêt pour l'aventure et le combat"
        ending_pos = current_text.find(target_ending)
        
        if ending_pos != -1:
            # Cut at the natural ending and add period
            fixed_text = current_text[:ending_pos + len(target_ending)] + "."
        else:
            # Fallback: just add a period
            fixed_text = current_text.rstrip() + "."
        
        # Count words
        word_count = len(fixed_text.split())
        
        print(f"📊 Fixed description:")
        print(f"   Word count: {word_count}")
        print(f"   Ends with: '{fixed_text[-50:]}'")
        
        # Clear and recreate
        db_manager.clear_collection(db_manager.config.descriptions_collection)
        
        # Create new description record
        description = DescriptionRecord(
            description_id="",
            asset_id=metadata['asset_id'],
            language=metadata['language'],
            description_text=fixed_text,
            model_name="manual-creation-fixed",
            generation_time=0.0,
            word_count=word_count,
            target_length=200,
            views_analyzed=["front", "back", "left", "right", "top", "iso_3d"]
        )
        
        # Add to database
        if db_manager.add_description(description):
            print("✅ Fixed description added to database")
            
            print(f"\n📝 Final description ({word_count} words):")
            print(f"{'='*60}")
            print(fixed_text)
            print(f"{'='*60}")
            
            return True
        else:
            print("❌ Failed to add fixed description")
            return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Fix error: {str(e)}", exc_info=True)
        return False

def main():
    """Main function"""
    
    success = fix_description_ending()
    
    if success:
        print("\n✅ Description ending fixed!")
        return 0
    else:
        print("\n❌ Failed to fix description ending!")
        return 1

if __name__ == "__main__":
    exit(main())
