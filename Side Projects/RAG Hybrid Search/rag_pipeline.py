import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uuid
import base64
from collections import Counter
import openai
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, list_collections, utility
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import streamlit as st

load_dotenv()

class RAGPipeline:
    def __init__(self, milvus_host="localhost", milvus_port="19530", collection_name="rag_assignment"):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.image_output_dir = "uploaded_images"
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
        self.prompt = hub.pull("rlm/rag-prompt")
        
        # Connect to Milvus
        self._connect_milvus()
        self._setup_collection()
        
        # Initialize BM25 retriever storage
        self.bm25_documents = []
        self.sparse_retriever = None
        
        # Create image directory
        os.makedirs(self.image_output_dir, exist_ok=True)
    
    def _connect_milvus(self):
        """Connect to Milvus database"""
        try:
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            print("Connected to Milvus successfully")
        except Exception as e:
            st.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _get_current_schema(self):
        """Get the current schema of the collection if it exists"""
        try:
            if self.collection_name in list_collections():
                collection = Collection(name=self.collection_name)
                return collection.schema
            return None
        except Exception as e:
            print(f"Error getting current schema: {e}")
            return None
    
    def _has_pdf_name_field(self):
        """Check if the current collection has the pdf_name field"""
        try:
            if self.collection_name in list_collections():
                collection = Collection(name=self.collection_name)
                field_names = [field.name for field in collection.schema.fields]
                return "pdf_name" in field_names
            return False
        except Exception as e:
            print(f"Error checking pdf_name field: {e}")
            return False
    
    def _migrate_collection(self):
        """Migrate existing collection to new schema with pdf_name field"""
        try:
            old_collection_name = f"{self.collection_name}_backup"
            
            # Rename old collection
            if self.collection_name in list_collections():
                print("Backing up old collection...")
                old_collection = Collection(name=self.collection_name)
                
                # Get all data from old collection
                old_data = []
                try:
                    results = old_collection.query(
                        expr="id != ''",
                        output_fields=["*"]
                    )
                    old_data = results
                except:
                    print("No data to migrate or query failed")
                
                # Drop old collection
                utility.drop_collection(self.collection_name)
                print(f"Dropped old collection: {self.collection_name}")
                
                # Create new collection with updated schema
                self._create_new_collection()
                
                # Migrate data if any exists
                if old_data:
                    print(f"Migrating {len(old_data)} records...")
                    self._migrate_data(old_data)
                
                return True
            else:
                # No existing collection, just create new one
                self._create_new_collection()
                return True
                
        except Exception as e:
            st.error(f"Error during migration: {e}")
            print(f"Migration error: {e}")
            return False
    
    def _create_new_collection(self):
        """Create new collection with updated schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=36),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=256),
        ]
        
        schema = CollectionSchema(fields, description="RAG Assignment with PDF tracking")
        self.collection = Collection(name=self.collection_name, schema=schema)
        self._create_index()
        print(f"Created new collection with updated schema: {self.collection_name}")
    
    def _migrate_data(self, old_data):
        """Migrate old data to new schema format"""
        try:
            migrated_docs = []
            
            for record in old_data:
                migrated_doc = {
                    "id": record.get("id", str(uuid.uuid4())),
                    "embedding": record.get("embedding", [0.0] * 1536),
                    "content": record.get("content", ""),
                    "type": record.get("type", "Composite"),
                    "section_title": record.get("section_title", ""),
                    "page_number": record.get("page_number", -1),
                    "image_path": record.get("image_path", ""),
                    "pdf_name": "legacy_document.pdf",
                }
                migrated_docs.append(migrated_doc)
            
            if migrated_docs:
                self.insert_documents(migrated_docs)
                print(f"Successfully migrated {len(migrated_docs)} records")
            
        except Exception as e:
            print(f"Error migrating data: {e}")
    
    def _setup_collection(self):
        """Setup Milvus collection schema with migration support"""
        try:
            if self.collection_name in list_collections():
                if self._has_pdf_name_field():
                    self.collection = Collection(name=self.collection_name)
                    print("Using existing collection with correct schema")
                else:
                    print("Collection exists but needs schema migration...")
                    if st.session_state.get('migration_confirmed', False) or self._confirm_migration():
                        self._migrate_collection()
                    else:
                        st.error("Migration required but not confirmed. Please restart the app and confirm migration.")
                        st.stop()
            else:
                self._create_new_collection()
            
            self.collection.load()
            
            # Initialize BM25 retriever with existing documents
            self._initialize_bm25_retriever()
            
        except Exception as e:
            st.error(f"Error setting up collection: {e}")
            raise
    
    def _confirm_migration(self):
        """Ask user to confirm migration"""
        if 'migration_confirmed' not in st.session_state:
            st.session_state.migration_confirmed = False
        
        if not st.session_state.migration_confirmed:
            st.warning("""
            ⚠️ **Schema Migration Required**
            
            Your existing collection needs to be updated to support PDF tracking features.
            This will:
            - Backup your current data
            - Create a new collection with updated schema
            - Migrate existing data (marked as 'legacy_document.pdf')
            
            **This is a one-time operation and is safe.**
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Proceed with Migration", type="primary"):
                    st.session_state.migration_confirmed = True
                    st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.stop()
            
            return False
        
        return True
    
    def _create_index(self):
        """Create vector index for the collection"""
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
    
    def _initialize_bm25_retriever(self):
        """Initialize BM25 retriever with existing documents"""
        try:
            # Get all documents from Milvus for BM25
            results = self.collection.query(
                expr="id != ''",
                output_fields=["id", "content", "type", "section_title", "page_number", "image_path", "pdf_name"]
            )
            
            self.bm25_documents = [
                Document(
                    page_content=result["content"],
                    metadata={
                        "id": result["id"],
                        "type": result["type"],
                        "section_title": result["section_title"],
                        "page_number": result["page_number"],
                        "image_path": result["image_path"],
                        "pdf_name": result["pdf_name"]
                    }
                )
                for result in results
            ]
            
            if self.bm25_documents:
                self.sparse_retriever = BM25Retriever.from_documents(self.bm25_documents)
                self.sparse_retriever.k = 3
                print(f"Initialized BM25 retriever with {len(self.bm25_documents)} documents")
            else:
                print("No documents found for BM25 initialization")
                
        except Exception as e:
            print(f"Error initializing BM25 retriever: {e}")
            self.sparse_retriever = None
    
    def _update_bm25_retriever(self):
        """Update BM25 retriever after new documents are added"""
        self._initialize_bm25_retriever()
    
    def check_pdf_exists(self, pdf_name):
        """Check if PDF already exists in the database"""
        try:
            results = self.collection.query(
                expr=f'pdf_name == "{pdf_name}"',
                output_fields=["id"],
                limit=1
            )
            return len(results) > 0
        except Exception as e:
            print(f"Error checking PDF existence: {e}")
            return False
    
    def get_pdf_statistics(self, pdf_name):
        """Get statistics for a specific PDF"""
        try:
            results = self.collection.query(
                expr=f'pdf_name == "{pdf_name}"',
                output_fields=["id", "page_number", "image_path"]
            )
            
            total_chunks = len(results)
            pages = set()
            total_images = 0
            
            for result in results:
                if result.get("page_number", -1) != -1:
                    pages.add(result["page_number"])
                
                image_paths = result.get("image_path", "")
                if image_paths:
                    total_images += len([p for p in image_paths.split(",") if p.strip()])
            
            return {
                "total_chunks": total_chunks,
                "total_pages": len(pages),
                "total_images": total_images
            }
        except Exception as e:
            print(f"Error getting PDF statistics: {e}")
            return {"total_chunks": 0, "total_pages": 0, "total_images": 0}
    
    def process_pdf(self, file_path, pdf_name):
        """Process PDF and extract chunks with metadata"""
        try:
            if self.check_pdf_exists(pdf_name):
                print(f"PDF {pdf_name} already exists in database")
                return []
            
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image", "Table"],
                image_output_dir_path=self.image_output_dir,
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000
            )
            
            documents = []
            
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                content_pieces, page_numbers, image_paths = [], [], []
                
                for idx, elem in enumerate(getattr(chunk.metadata, 'orig_elements', [])):
                    if (text := getattr(elem, 'text', None)):
                        content_pieces.append(text)
                    
                    if (page := getattr(elem.metadata, 'page_number', None)) is not None:
                        page_numbers.append(page)
                    
                    if (b64 := getattr(elem.metadata, 'image_base64', None)):
                        try:
                            img_bytes = base64.b64decode(b64)
                            img_path = os.path.join(self.image_output_dir, f"{pdf_name}_{chunk_id}_{idx}.png")
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            image_paths.append(img_path)
                        except Exception as e:
                            print(f"Image decode error in chunk {chunk_id}: {e}")
                
                content = "\n".join(content_pieces)
                if not content.strip():
                    continue
                    
                page_number = Counter(page_numbers).most_common(1)[0][0] if page_numbers else -1
                
                try:
                    embedding_response = openai.embeddings.create(
                        input=content,
                        model="text-embedding-3-small"
                    )
                    embedding = embedding_response.data[0].embedding
                except Exception as e:
                    print(f"Embedding error in chunk {chunk_id}: {e}")
                    embedding = [0.0] * 1536
                
                doc = {
                    "id": chunk_id,
                    "embedding": embedding,
                    "content": content,
                    "type": getattr(chunk, 'type', "Composite"),
                    "section_title": getattr(chunk, 'section_title', "") or "",
                    "page_number": page_number,
                    "image_path": image_paths,
                    "pdf_name": pdf_name,
                }
                
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            st.error(f"Error processing PDF {pdf_name}: {e}")
            return []
    
    def insert_documents(self, documents):
        """Insert documents into Milvus collection"""
        if not documents:
            return False
        
        try:
            data_to_insert = [
                [doc["id"] for doc in documents],
                [doc["embedding"] for doc in documents],
                [doc["content"][:65000] for doc in documents],
                [doc["type"] for doc in documents],
                [doc["section_title"][:256] for doc in documents],
                [doc["page_number"] for doc in documents],
                [",".join(doc["image_path"])[:1024] for doc in documents],
                [doc["pdf_name"] for doc in documents],
            ]
            
            self.collection.insert(data_to_insert)
            self.collection.flush()
            
            # Update BM25 retriever with new documents
            self._update_bm25_retriever()
            
            print(f"Successfully inserted {len(documents)} documents")
            return True
            
        except Exception as e:
            st.error(f"Error inserting documents: {e}")
            return False
    
    def delete_pdf_records(self, pdf_name):
        """Delete all records associated with a specific PDF"""
        try:
            # Get all image paths for this PDF
            results = self.collection.query(
                expr=f'pdf_name == "{pdf_name}"',
                output_fields=["image_path"]
            )
            
            # Collect image paths to delete
            images_to_delete = []
            for result in results:
                image_paths = result.get("image_path", "")
                if image_paths:
                    paths = [p.strip() for p in image_paths.split(",") if p.strip()]
                    images_to_delete.extend(paths)
            
            # Delete records from Milvus
            expr = f'pdf_name == "{pdf_name}"'
            self.collection.delete(expr)
            self.collection.flush()
            
            # Update BM25 retriever after deletion
            self._update_bm25_retriever()
            
            # Delete image files
            deleted_images = 0
            for img_path in images_to_delete:
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        deleted_images += 1
                except Exception as e:
                    print(f"Error deleting image {img_path}: {e}")
            
            # Clean up orphaned images
            if os.path.exists(self.image_output_dir):
                for filename in os.listdir(self.image_output_dir):
                    if filename.startswith(f"{pdf_name}_"):
                        try:
                            os.remove(os.path.join(self.image_output_dir, filename))
                            deleted_images += 1
                        except Exception as e:
                            print(f"Error deleting orphaned image {filename}: {e}")
            
            print(f"Successfully deleted PDF '{pdf_name}': {len(results)} records and {deleted_images} images")
            return True
            
        except Exception as e:
            st.error(f"Error deleting PDF records for '{pdf_name}': {e}")
            return False
    
    def get_stored_pdfs(self):
        """Get list of all stored PDFs with their statistics"""
        try:
            results = self.collection.query(
                expr="pdf_name != ''",
                output_fields=["pdf_name"]
            )
            
            pdf_names = list(set([result["pdf_name"] for result in results]))
            
            pdf_info = []
            for pdf_name in pdf_names:
                stats = self.get_pdf_statistics(pdf_name)
                pdf_info.append({
                    "name": pdf_name,
                    "chunks": stats["total_chunks"],
                    "pages": stats["total_pages"],
                    "images": stats["total_images"]
                })
            
            return sorted(pdf_info, key=lambda x: x["name"])
            
        except Exception as e:
            st.error(f"Error getting stored PDFs: {e}")
            return []
    
    def search_and_rank(self, query, k=3, pdf_filter=None):
        """Hybrid search and rank documents using ensemble retriever with optional PDF filtering"""
        try:
            # Create dense vector retriever
            vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"host": self.milvus_host, "port": self.milvus_port},
                text_field="content",
                vector_field="embedding",
                index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"},
            )
            
            search_kwargs = {"k": k}
            if pdf_filter:
                search_kwargs["expr"] = f'pdf_name == "{pdf_filter}"'
            
            dense_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            
            # Create sparse retriever (BM25) with filtering if needed
            if self.sparse_retriever is None:
                # st.warning("BM25 retriever not initialized. Using dense retrieval only.")
                top_k_docs = dense_retriever.get_relevant_documents(query)
            else:
                # Filter BM25 documents if PDF filter is specified
                filtered_bm25_docs = self.bm25_documents
                if pdf_filter:
                    filtered_bm25_docs = [
                        doc for doc in self.bm25_documents 
                        if doc.metadata.get("pdf_name") == pdf_filter
                    ]
                
                if not filtered_bm25_docs:
                    st.warning(f"No documents found for PDF filter: {pdf_filter}")
                    return []
                
                # Create filtered BM25 retriever
                filtered_sparse_retriever = BM25Retriever.from_documents(filtered_bm25_docs)
                filtered_sparse_retriever.k = k
                
                # Create ensemble retriever with weighted combination
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[filtered_sparse_retriever, dense_retriever],
                    weights=[0.6, 0.4]  # Favor sparse (BM25) over dense
                )
                
                top_k_docs = ensemble_retriever.invoke(query)
            
            if not top_k_docs:
                return []
            
            # Re-rank using cross-encoder
            pairs = [(query, doc.page_content) for doc in top_k_docs]
            scores = self.cross_encoder.predict(pairs)
            
            scored_docs = sorted(zip(top_k_docs, scores), key=lambda x: x[1], reverse=True)
            ranked_docs = [doc for doc, score in scored_docs]
            
            return ranked_docs
            
        except Exception as e:
            st.error(f"Error in hybrid search and ranking: {e}")
            return []
    
    def generate_answer(self, query, ranked_docs, num_docs=2):
        """Generate answer using RAG with top ranked documents"""
        try:
            if not ranked_docs:
                return "No relevant documents found for your query.", []
            
            top_docs = ranked_docs[:num_docs]
            context = "\n\n".join([doc.page_content for doc in top_docs])
            
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            response = chain.run({"context": context, "question": query})
            
            return response, top_docs
            
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            return "Error generating response.", []
    
    def get_collection_stats(self):
        """Get overall collection statistics"""
        try:
            total_entities = self.collection.num_entities
            
            results = self.collection.query(
                expr="pdf_name != ''",
                output_fields=["pdf_name"]
            )
            unique_pdfs = len(set([result["pdf_name"] for result in results]))
            
            return {
                "total_chunks": total_entities,
                "total_pdfs": unique_pdfs
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"total_chunks": 0, "total_pdfs": 0}