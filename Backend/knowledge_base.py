import chromadb
import os
from typing import List, Dict, Any
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UniversityKnowledgeBase:
    def __init__(self):
        self.host = Config.CHROMA_HOST
        self.port = Config.CHROMA_PORT
        self.client = None
        self.collection = None

        try:
            # Initialize ChromaDB client
            logger.info(f"🔗 Connecting to ChromaDB at {self.host}:{self.port}")
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port
            )
            
            # Test connection
            self.client.heartbeat()
            logger.info("✅ ChromaDB connection successful")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=Config.COLLECTION_NAME,
                metadata={"description": "University knowledge base for chatbot"}
            )
            logger.info(f"✅ Collection '{Config.COLLECTION_NAME}' ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            raise

    def load_knowledge_files(self) -> List[Dict[str, Any]]:
        """Load all knowledge base files from folder structure"""
        documents = []

        if not os.path.exists(Config.KNOWLEDGE_PATH):
            logger.warning(f"Knowledge path {Config.KNOWLEDGE_PATH} not found!")
            return documents

        doc_id = 0
        logger.info(f"📂 Loading knowledge files from: {Config.KNOWLEDGE_PATH}")

        # Walk through all subdirectories
        for root, dirs, files in os.walk(Config.KNOWLEDGE_PATH):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)

                    # Extract category and subcategory from folder structure
                    relative_path = os.path.relpath(file_path, Config.KNOWLEDGE_PATH)
                    path_parts = relative_path.split(os.sep)

                    # Main category is the first folder
                    category = path_parts[0] if len(path_parts) > 1 else 'general'

                    # Subcategory is the filename without extension
                    subcategory = file.replace('.txt', '')

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()

                            if content:
                                documents.append({
                                    'category': category,
                                    'subcategory': subcategory,
                                    'content': content,
                                    'source': 'knowledge_base',
                                    'filename': file,
                                    'filepath': relative_path,
                                    'doc_id': doc_id
                                })
                                logger.info(f"📖 Loaded: {category}/{subcategory} ({len(content)} chars)")
                                doc_id += 1
                            else:
                                logger.warning(f"Empty file: {relative_path}")
                    except Exception as e:
                        logger.error(f"Error loading {relative_path}: {e}")

        logger.info(f"✅ Loaded {len(documents)} knowledge base files")
        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks"""
        chunked_documents = []
        logger.info("✂️ Chunking documents...")

        for doc in documents:
            content = doc['content']
            
            # Skip very short documents
            if len(content) < 50:
                chunked_documents.append({
                    'content': content,
                    'category': doc['category'],
                    'subcategory': doc.get('subcategory', ''),
                    'source': doc['source'],
                    'filename': doc['filename'],
                    'filepath': doc.get('filepath', ''),
                    'chunk_id': len(chunked_documents)
                })
                continue

            # Simple chunking by sentences
            sentences = content.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk + sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunked_documents.append({
                            'content': current_chunk.strip(),
                            'category': doc['category'],
                            'subcategory': doc.get('subcategory', ''),
                            'source': doc['source'],
                            'filename': doc['filename'],
                            'filepath': doc.get('filepath', ''),
                            'chunk_id': len(chunked_documents)
                        })
                    current_chunk = sentence + ". "

            # Add the last chunk
            if current_chunk.strip():
                chunked_documents.append({
                    'content': current_chunk.strip(),
                    'category': doc['category'],
                    'subcategory': doc.get('subcategory', ''),
                    'source': doc['source'],
                    'filename': doc['filename'],
                    'filepath': doc.get('filepath', ''),
                    'chunk_id': len(chunked_documents)
                })

        logger.info(f"📄 Split into {len(chunked_documents)} chunks")
        return chunked_documents

    def setup_knowledge_base(self):
        """Setup and populate the knowledge base"""
        logger.info("🔄 Setting up knowledge base...")

        # Load documents
        documents = self.load_knowledge_files()
        if not documents:
            logger.error("❌ No documents found to add to knowledge base")
            return False

        # Chunk documents
        chunked_docs = self.chunk_documents(documents)

        # Add to ChromaDB
        success = self.add_documents(chunked_docs)

        # Verify setup
        stats = self.get_stats()
        logger.info(f"📊 Knowledge base setup complete: {stats}")

        return success

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to ChromaDB"""
        try:
            all_documents = []
            all_metadatas = []
            all_ids = []

            for i, doc in enumerate(documents):
                all_documents.append(doc['content'])
                all_metadatas.append({
                    'category': doc['category'],
                    'subcategory': doc.get('subcategory', ''),
                    'source': doc['source'],
                    'filename': doc['filename'],
                    'filepath': doc.get('filepath', ''),
                    'chunk_id': doc['chunk_id']
                })
                all_ids.append(
                    f"{doc['category']}_{doc.get('subcategory', 'general')}_{i}_{doc['chunk_id']}"
                )

            # Add to collection in batches
            batch_size = 50
            total_added = 0
            
            for i in range(0, len(all_documents), batch_size):
                end_idx = min(i + batch_size, len(all_documents))
                batch_docs = all_documents[i:end_idx]
                batch_metadata = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]

                if batch_docs:
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metadata,
                        ids=batch_ids
                    )
                    total_added += len(batch_docs)
                    logger.info(f"✅ Added batch {i//batch_size + 1}: {len(batch_docs)} chunks")

            logger.info(f"🎉 Successfully added {total_added} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to ChromaDB: {e}")
            return False

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            logger.info(f"🔍 Searching for: '{query}' (n_results: {n_results})")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (higher is better)
                    similarity = 1.0 / (1.0 + distance) if distance else 0.8

                    formatted_results.append({
                        'content': doc,
                        'category': metadata['category'],
                        'subcategory': metadata.get('subcategory', ''),
                        'similarity_score': round(similarity, 3),
                        'distance': distance,
                        'metadata': metadata
                    })
                    
                    logger.info(f"   📄 Result {i+1}: {doc[:100]}... (similarity: {similarity:.3f})")

            logger.info(f"✅ Search completed: {len(formatted_results)} results found")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "status": "connected",
                "host": self.host,
                "port": self.port,
                "collection": Config.COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {"status": "error", "error": str(e)}


def setup_knowledge_base():
    """Convenience function to setup knowledge base"""
    try:
        kb = UniversityKnowledgeBase()
        success = kb.setup_knowledge_base()
        return success
    except Exception as e:
        logger.error(f"❌ Failed to setup knowledge base: {e}")
        return False


if __name__ == "__main__":
    success = setup_knowledge_base()
    if success:
        print("\n🎉 Knowledge base setup completed successfully!")
        
        # Test search
        kb = UniversityKnowledgeBase()
        test_queries = [
            "admission process",
            "MCA fees", 
            "hostel facilities"
        ]
        
        print("\n🧪 Testing search functionality:")
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = kb.search(query, n_results=1)
            if results:
                print(f"   ✅ Found: {results[0]['content'][:100]}...")
            else:
                print("   ❌ No results found")
    else:
        print("\n❌ Knowledge base setup failed!")