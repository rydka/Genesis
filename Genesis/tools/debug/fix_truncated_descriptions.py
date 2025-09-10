#!/usr/bin/env python3
"""
Script to fix truncated descriptions by regenerating them with improved parameters
"""

import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_truncated_descriptions():
    """Fix truncated descriptions by regenerating them"""
    
    print("🔧 Fixing Truncated Descriptions")
    print("=" * 60)
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        from fantasy_rag.description.llm_interface import OllamaVisionInterface
        from fantasy_rag.description.prompt_templates import PromptContext, DescriptionLanguage
        from fantasy_rag.database.schema_definitions import DescriptionRecord
        
        # Initialize database
        print("🔧 Initializing database...")
        config = DatabaseConfig(db_path="ressources/data/fantasy_rag_db")
        db_manager = ChromaDBManager(config)
        
        if not db_manager.initialize():
            print("❌ Failed to initialize database")
            return False
        
        print("✅ Database initialized")
        
        # Initialize vision interface with improved parameters
        print("🔧 Initializing vision interface with improved parameters...")
        interface = OllamaVisionInterface(
            model_name="llama3.2-vision:latest",
            max_tokens=2000,  # Increased from 1000
            temperature=0.7,
            timeout=180  # Increased timeout
        )
        
        if not interface.verify_connection():
            print("❌ Failed to connect to Ollama")
            return False
        
        print("✅ Connected to Ollama with improved parameters")
        print(f"   Max tokens: {interface.max_tokens}")
        print(f"   Timeout: {interface.timeout}s")
        
        # Get existing descriptions and check for truncation
        desc_collection = db_manager.collections[db_manager.config.descriptions_collection]
        all_docs = desc_collection.get()
        
        if not all_docs['ids']:
            print("📝 No descriptions found in database")
            return True
        
        print(f"📝 Found {len(all_docs['ids'])} descriptions")
        
        # Identify truncated descriptions
        truncated_descriptions = []
        for i, doc_id in enumerate(all_docs['ids']):
            metadata = all_docs['metadatas'][i]
            document = all_docs['documents'][i]
            
            # Check if truncated
            is_truncated = (
                not document.strip().endswith(('.', '!', '?')) or
                'évident' in document[-20:] or
                len(document.split()) < metadata.get('target_length', 200) * 0.7  # Less than 70% of target
            )
            
            if is_truncated:
                truncated_descriptions.append({
                    'id': doc_id,
                    'metadata': metadata,
                    'document': document,
                    'asset_id': metadata.get('asset_id'),
                    'language': metadata.get('language'),
                    'current_words': len(document.split())
                })
        
        if not truncated_descriptions:
            print("✅ No truncated descriptions found!")
            return True
        
        print(f"🔧 Found {len(truncated_descriptions)} truncated descriptions to fix")
        
        # Process each truncated description
        fixed_count = 0
        for i, desc_info in enumerate(truncated_descriptions, 1):
            print(f"\n🔧 Fixing {i}/{len(truncated_descriptions)}: {desc_info['asset_id']}")
            print(f"   Current: {desc_info['current_words']} words (truncated)")
            print(f"   Language: {desc_info['language']}")
            
            # Get asset info
            asset = db_manager.get_asset(desc_info['asset_id'])
            if not asset:
                print(f"   ❌ Asset not found: {desc_info['asset_id']}")
                continue
            
            # Get renders
            renders = db_manager.get_asset_renders(desc_info['asset_id'])
            if not renders:
                print(f"   ❌ No renders found for: {desc_info['asset_id']}")
                continue
            
            # Create view paths
            view_paths = {}
            for render in renders:
                render_path = Path(render.image_path)
                if render_path.exists():
                    view_paths[render.view_name] = render_path
            
            if not view_paths:
                print(f"   ❌ No valid render files found")
                continue
            
            print(f"   🖼️ Using {len(view_paths)} views")
            
            # Create context for 200 words with special instructions
            context = PromptContext(
                asset_name=asset.original_filename.replace('.stl', '').replace('-', ' '),
                view_count=len(view_paths),
                target_length=200,  # Target 200 words
                language=DescriptionLanguage.FRENCH if desc_info['language'] == 'français' else DescriptionLanguage.ENGLISH,
                special_instructions="""
IMPORTANT: Vous DEVEZ terminer votre description par une phrase complète avec un point final.
Ne vous arrêtez pas au milieu d'une phrase. Assurez-vous que la description est complète et bien structurée.
Visez exactement 200 mots avec une conclusion appropriée.
"""
            )
            
            # Generate new description
            print(f"   🎯 Generating complete 200-word description...")
            start_time = time.time()
            
            response = interface.generate_description(view_paths, context)
            generation_time = time.time() - start_time
            
            if response.success:
                # Check if the new description is complete
                is_complete = response.description.strip().endswith(('.', '!', '?'))
                word_count = response.word_count
                
                print(f"   ✅ Generated: {word_count} words in {generation_time:.1f}s")
                print(f"   Complete: {'✅ Yes' if is_complete else '❌ Still truncated'}")
                
                if is_complete:
                    # Create new description record
                    new_description = DescriptionRecord(
                        description_id="",  # Will be auto-generated
                        asset_id=desc_info['asset_id'],
                        language=desc_info['language'],
                        description_text=response.description,
                        model_name="llama3.2-vision:latest",
                        generation_time=response.processing_time,
                        word_count=response.word_count,
                        target_length=200,
                        views_analyzed=list(view_paths.keys())
                    )
                    
                    # Update in database
                    if db_manager.add_description(new_description):
                        fixed_count += 1
                        print(f"   ✅ Fixed and updated in database")
                        
                        # Show improvement
                        print(f"   📈 Improvement: {desc_info['current_words']} → {word_count} words")
                        print(f"   📝 Ends with: '{response.description.strip()[-50:]}'")
                    else:
                        print(f"   ❌ Failed to update database")
                else:
                    print(f"   ⚠️ Description still appears truncated, skipping")
            else:
                print(f"   ❌ Generation failed: {response.error_message}")
            
            # Delay between requests
            time.sleep(3)
        
        print(f"\n" + "=" * 60)
        print(f"📊 Fix Summary:")
        print(f"   Truncated descriptions found: {len(truncated_descriptions)}")
        print(f"   Successfully fixed: {fixed_count}")
        print(f"   Success rate: {fixed_count/len(truncated_descriptions)*100:.1f}%")
        
        if fixed_count > 0:
            print(f"\n🎉 Successfully fixed {fixed_count} truncated descriptions!")
            print(f"\n💡 Next steps:")
            print(f"   1. Test search: python test_epee_search.py")
            print(f"   2. Check web interface: python -m fantasy_rag.web.app")
            print(f"   3. Verify descriptions: python debug_description_truncation.py")
        
        return fixed_count > 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Fix error: {str(e)}", exc_info=True)
        return False

def main():
    """Main function"""
    
    success = fix_truncated_descriptions()
    
    if success:
        print("\n✅ Truncated descriptions fixed successfully!")
        return 0
    else:
        print("\n❌ Failed to fix truncated descriptions!")
        return 1

if __name__ == "__main__":
    exit(main())
