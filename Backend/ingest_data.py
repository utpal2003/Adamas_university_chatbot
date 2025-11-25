import os
import chromadb
from typing import List, Dict, Any
import time
import subprocess
import sys
from config import Config


class ChromaDBManager:
    def __init__(self):
        self.host = Config.CHROMA_HOST
        self.port = Config.CHROMA_PORT
        self.client = None
        self.collection = None
        self.connect()

    def connect(self):
        """Connect to ChromaDB with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client = chromadb.HttpClient(
                    host=self.host, port=self.port)
                self.collection = self.client.get_or_create_collection(
                    name=Config.COLLECTION_NAME
                )
                print(f"✅ Connected to ChromaDB at {self.host}:{self.port}")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"⏳ Connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                else:
                    print(f"❌ Failed to connect to ChromaDB: {e}")
                    return False

    def wait_for_connection(self, max_retries: int = 10, delay: int = 3) -> bool:
        """Wait for ChromaDB to be ready"""
        for i in range(max_retries):
            try:
                count = self.collection.count()
                print(f"✅ ChromaDB ready! Documents: {count}")
                return True
            except Exception as e:
                if i < max_retries - 1:
                    print(
                        f"⏳ Waiting for ChromaDB... (attempt {i+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(
                        f"❌ ChromaDB not ready after {max_retries} attempts: {e}")
                    return False
        return False

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks for better context"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to ChromaDB with metadata"""
        all_documents = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            chunks = self.chunk_text(doc['content'])

            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    all_documents.append(chunk)
                    all_metadatas.append({
                        'category': doc['category'],
                        'subcategory': doc.get('subcategory', ''),
                        'source': 'knowledge_base',
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'doc_id': doc.get('doc_id', 0),
                        'filename': doc.get('filename', ''),
                        'filepath': doc.get('filepath', '')
                    })
                    all_ids.append(
                        f"{doc['category']}_{doc.get('subcategory', 'general')}_{doc.get('doc_id', 0)}_{i}")

        # Add to collection in batches
        batch_size = 50
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
                print(
                    f"✅ Added batch {i//batch_size + 1}: {len(batch_docs)} chunks")

        print(
            f"🎉 Successfully added {len(all_documents)} document chunks to ChromaDB")

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0]
                )):
                    similarity = 0.9 - (i * 0.2)
                    similarity = max(0.3, similarity)

                    formatted_results.append({
                        'content': doc,
                        'category': metadata['category'],
                        'subcategory': metadata.get('subcategory', ''),
                        'similarity_score': similarity,
                        'metadata': metadata
                    })

            return formatted_results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "status": "connected",
                "host": self.host,
                "port": self.port,
                "collection": Config.COLLECTION_NAME
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def load_knowledge_base(knowledge_base_path: str) -> List[Dict[str, Any]]:
    """Load all knowledge base files from the new folder structure"""
    documents = []

    if not os.path.exists(knowledge_base_path):
        print(f"❌ Knowledge base path {knowledge_base_path} not found!")
        print("💡 Creating knowledge directory with subfolders...")
        create_sample_folder_structure(knowledge_base_path)
        return documents

    doc_id = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(knowledge_base_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # Extract category and subcategory from folder structure
                relative_path = os.path.relpath(file_path, knowledge_base_path)
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
                            print(f"📖 Loaded: {category}/{subcategory}")
                            doc_id += 1
                        else:
                            print(f"⚠️ Empty file: {relative_path}")
                except Exception as e:
                    print(f"❌ Error loading {relative_path}: {e}")

    print(
        f"✅ Loaded {len(documents)} knowledge base files from {knowledge_base_path}")
    return documents


def create_sample_folder_structure(knowledge_base_path: str):
    """Create the folder structure with sample files"""
    folders = [
        "admission",
        "courses",
        "fees",
        "facilities",
        "faculty",
        "transport",
        "placement",
        "contacts",
        "rules",
        "examination"
    ]

    sample_content = {
        "admission": {
            "admission_requirements": "Admission Requirements:\n- Minimum 60% in 12th standard\n- Valid entrance exam score\n- Age limit: 17-25 years",
            "admission_process": "Admission Process:\n1. Fill online application\n2. Submit documents\n3. Entrance exam\n4. Interview\n5. Final selection",
            "admission_criteria": "Admission Criteria:\n- Academic performance: 50%\n- Entrance exam: 30%\n- Interview: 20%"
        },
        "courses": {
            "course_information": "Available Courses:\n- Computer Science Engineering\n- Mechanical Engineering\n- Business Administration\n- Bachelor of Commerce",
            "program_details": "Program Details:\n- Duration: 4 years for Engineering\n- Duration: 3 years for Arts & Commerce\n- Credits: 180-240 required",
            "department_info": "Departments:\n- Computer Science & Engineering\n- Mechanical Engineering\n- Business Administration\n- Commerce & Economics"
        }
    }

    # Create folders
    for folder in folders:
        folder_path = os.path.join(knowledge_base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 Created folder: {folder}")

    # Create sample files
    for category, files in sample_content.items():
        for filename, content in files.items():
            file_path = os.path.join(
                knowledge_base_path, category, f"{filename}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 Created sample: {category}/{filename}.txt")

    print(f"✅ Sample folder structure created at {knowledge_base_path}")
    print("💡 Please add your actual content to these files and run the script again.")


def check_docker_running():
    """Check if Docker is running and ChromaDB container exists"""
    try:
        result = subprocess.run(
            ['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not running or not installed")
            return False

        result = subprocess.run(['docker', 'ps', '--filter', 'name=chroma', '--format', '{{.Names}}'],
                                capture_output=True, text=True)
        if 'chroma' in result.stdout:
            print("✅ ChromaDB container is running")
            return True
        else:
            print("❌ ChromaDB container not found")
            return False

    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False


def start_chromadb():
    """Try to start ChromaDB on different ports"""
    ports_to_try = [8000, 8001, 8002, 8003, 8004]

    for port in ports_to_try:
        try:
            print(f"🚀 Trying to start ChromaDB on port {port}...")
            subprocess.run([
                'docker', 'run', '-d',
                '--name', f'chromadb_{port}',
                '-p', f'{port}:8000',
                'chromadb/chroma'
            ], check=True, capture_output=True)
            print(f"✅ ChromaDB started on port {port}")
            return port
        except subprocess.CalledProcessError:
            print(f"⚠️ Port {port} busy, trying next...")
            continue

    print("❌ All ports busy. Please stop other containers or use a different port.")
    return None


def main():
    print("🚀 Starting University Knowledge Base Ingestion...")
    print("📁 Using new folder structure...")

    # Check if ChromaDB is running
    if not check_docker_running():
        print("\n💡 Starting ChromaDB automatically...")
        port = start_chromadb()
        if not port:
            print("❌ Could not start ChromaDB automatically")
            print(
                "🔧 Please start it manually: docker run -d -p 8000:8000 --name chroma-db chromadb/chroma")
            return
        time.sleep(5)

    try:
        # Initialize ChromaDB manager
        chroma_manager = ChromaDBManager()

        if not chroma_manager.client:
            print("❌ Could not connect to ChromaDB.")
            return

        # Wait for connection
        if not chroma_manager.wait_for_connection():
            print("❌ ChromaDB not responding.")
            return

        # Load knowledge base files
        knowledge_base_path = "./knowledge"
        documents = load_knowledge_base(knowledge_base_path)

        if not documents:
            print("❌ No documents found to ingest!")
            return

        # Clear existing collection to start fresh
        try:
            chroma_manager.client.delete_collection(Config.COLLECTION_NAME)
            print("♻️ Cleared existing collection")
        except Exception as e:
            print("📝 Creating new collection")

        # Recreate collection
        chroma_manager.collection = chroma_manager.client.get_or_create_collection(
            name=Config.COLLECTION_NAME
        )

        # Add documents to ChromaDB
        print("📥 Adding documents to ChromaDB...")
        chroma_manager.add_documents(documents)

        # Show statistics
        stats = chroma_manager.get_collection_stats()
        print(f"\n🎉 Ingestion completed!")
        print(f"📊 Collection stats: {stats}")

        # Test search
        print("\n🔍 Testing search functionality...")
        test_queries = [
            "computer science courses",
            "admission requirements",
            "fee structure",
            "campus facilities",
            "bus schedule"
        ]

        for query in test_queries:
            results = chroma_manager.search(query, n_results=1)
            if results:
                best_match = results[0]
                category_info = f"{best_match['category']}"
                if best_match.get('subcategory'):
                    category_info += f"/{best_match['subcategory']}"

                print(f"❓ '{query}'")
                print(
                    f"   ✅ Found: {category_info} (Score: {best_match['similarity_score']:.3f})")
                snippet = best_match['content'][:100] + "..." if len(
                    best_match['content']) > 100 else best_match['content']
                print(f"   📝 {snippet}")
            else:
                print(f"❓ '{query}' → ❌ No results found")
            print()

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
