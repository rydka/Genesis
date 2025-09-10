#!/usr/bin/env python3
"""
Script to check current descriptions in the database
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_descriptions():
    """Check current descriptions in database"""
    
    print("📋 Checking Current Descriptions")
    print("=" * 50)
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        
        # Initialize database
        config = DatabaseConfig(db_path="ressources/data/fantasy_rag_db")
        db_manager = ChromaDBManager(config)
        
        if not db_manager.initialize():
            print("❌ Failed to initialize database")
            return False
        
        print("✅ Database initialized")
        
        # Get descriptions directly from the collection
        try:
            # Access the descriptions collection directly
            desc_collection = db_manager.collections[db_manager.config.descriptions_collection]

            # Get all description documents
            all_docs = desc_collection.get()

            if not all_docs['ids']:
                print("📝 No descriptions found in database")
                return True

            # Convert to DescriptionRecord objects
            from fantasy_rag.database.schema_definitions import DescriptionRecord
            descriptions = []

            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i]
                document = all_docs['documents'][i]

                desc = DescriptionRecord(
                    description_id=doc_id,
                    asset_id=metadata.get('asset_id', ''),
                    language=metadata.get('language', ''),
                    description_text=document,
                    model_name=metadata.get('model_name', ''),
                    generation_time=metadata.get('generation_time', 0.0),
                    word_count=metadata.get('word_count', 0),
                    target_length=metadata.get('target_length', 0),
                    views_analyzed=metadata.get('views_analyzed', [])
                )
                descriptions.append(desc)

        except Exception as e:
            print(f"❌ Error accessing descriptions: {str(e)}")
            return False

        if not descriptions:
            print("📝 No descriptions found in database")
            return True
        
        print(f"📝 Found {len(descriptions)} descriptions:")
        print()
        
        # Analyze descriptions
        total_words = 0
        target_200_count = 0
        within_range_count = 0
        
        for i, desc in enumerate(descriptions, 1):
            print(f"{i}. Asset: {desc.asset_id}")
            print(f"   Language: {desc.language}")
            print(f"   Word count: {desc.word_count} words")
            print(f"   Target length: {desc.target_length} words")
            print(f"   Model: {desc.model_name}")
            
            # Calculate difference from 200 words
            diff_from_200 = abs(desc.word_count - 200)
            percentage_diff = (diff_from_200 / 200) * 100
            
            print(f"   Difference from 200: {diff_from_200} words ({percentage_diff:.1f}%)")
            
            # Check if within acceptable range (±20 words = ±10%)
            if diff_from_200 <= 20:
                print(f"   Status: ✅ Within range")
                within_range_count += 1
            else:
                print(f"   Status: ❌ Needs updating")
            
            # Check target length
            if desc.target_length == 200:
                target_200_count += 1
            
            total_words += desc.word_count
            
            # Show preview
            preview = desc.description_text[:100].replace('\n', ' ')
            print(f"   Preview: {preview}...")
            print()
        
        # Summary
        print("=" * 50)
        print("📊 Summary:")
        print(f"   Total descriptions: {len(descriptions)}")
        print(f"   Average word count: {total_words/len(descriptions):.1f} words")
        print(f"   Target 200 words: {target_200_count}/{len(descriptions)}")
        print(f"   Within ±20 words of 200: {within_range_count}/{len(descriptions)}")
        print(f"   Need updating: {len(descriptions) - within_range_count}")
        
        if within_range_count == len(descriptions):
            print("\n🎉 All descriptions are within acceptable range!")
        else:
            print(f"\n🔄 {len(descriptions) - within_range_count} descriptions need updating")
            print("   Run: python regenerate_descriptions_200_words.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Check error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    check_descriptions()
