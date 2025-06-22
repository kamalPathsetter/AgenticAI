# RAG PDF Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides intelligent question-answering with **hybrid search** capabilities and visual context.

## 🌟 Features

### Core Capabilities
- **📚 PDF Processing**: Advanced document parsing with text, image, and table extraction
- **🔄 Hybrid Search**: Combines BM25 (sparse) and vector search (dense) with cross-encoder reranking
- **🖼️ Visual Context**: Automatic image extraction and display from relevant document sections
- **📊 Multi-PDF Management**: Upload, track, and manage multiple PDF documents
- **⚙️ Configurable Search**: Adjustable parameters for search scope and result count

### Technical Architecture
- **Vector Database**: Milvus for scalable vector storage and retrieval
- **Embeddings**: OpenAI text-embedding-3-small for semantic understanding
- **LLM**: GPT-4o-mini for answer generation
- **Search Strategy**: 
  - BM25 retriever (60% weight) for keyword matching
  - Vector search (40% weight) for semantic similarity
  - Cross-encoder reranking for optimal relevance

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (for Milvus)
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-pdf-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Milvus Database**
```bash
# Using Docker Compose
docker-compose up -d

# Or using standalone Docker
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest standalone
```

4. **Environment Setup**
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_TRACING_V2=true
```

5. **Run the Application**
```bash
streamlit run app.py
```

## 📖 Usage

### Upload PDFs
1. Use the sidebar to upload one or more PDF files
2. The system automatically processes and extracts:
   - Text content with intelligent chunking
   - Images and diagrams
   - Tables and structured data
3. Documents are stored with metadata tracking

### Ask Questions
1. Enter your question in the main interface
2. Adjust search parameters:
   - **Documents to retrieve**: Breadth of search (1-10)
   - **Documents for answer**: Depth of analysis (1-5)
   - **PDF filter**: Search within specific documents
3. Click "Hybrid Search & Answer" to get results

### View Results
- **Answer**: AI-generated response based on retrieved content
- **Sources**: Ranked source documents with metadata
- **Visual Context**: Related images and diagrams
- **Statistics**: Retrieval metrics and search methodology

## 🏗️ Architecture

### Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   RAG Pipeline   │────│   Milvus DB     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌──────────────────┐
                    │  Hybrid Search   │
                    │  • BM25 (60%)    │
                    │  • Vector (40%)  │
                    │  • Reranking     │
                    └──────────────────┘
```

### File Structure
```
rag-pdf-assistant/
├── app.py                 # Streamlit web interface
├── rag_pipeline.py        # Core RAG implementation
├── rag_v2.ipynb          # Development notebook with hybrid search
├── rag.ipynb             # Basic RAG implementation
├── uploaded_images/       # Extracted images storage
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Search Parameters
- **Retrieval Count**: Number of documents to retrieve (default: 5)
- **Answer Documents**: Documents used for final answer (default: 2)
- **PDF Filter**: Restrict search to specific documents
- **Hybrid Weights**: BM25 (0.6) + Vector (0.4) balance

### System Settings
- **Chunk Size**: 10,000 characters with 2,000 overlap
- **Vector Dimensions**: 1536 (OpenAI embedding size)
- **Cross-encoder**: ms-marco-MiniLM-L6-v2 for reranking
- **Index Type**: IVF_FLAT with COSINE similarity

## 📊 Performance

### Search Methods Comparison
| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| BM25 Only | Fast | Good for keywords | Exact term matching |
| Vector Only | Medium | Good for semantics | Conceptual queries |
| Hybrid | Medium | Best overall | Balanced retrieval |

### Scalability
- **Documents**: Tested with 100+ PDFs
- **Chunks**: Handles 10,000+ text segments
- **Images**: Automatic extraction and storage
- **Memory**: Efficient with lazy loading

## 🛠️ Development

### Key Classes
- **`RAGPipeline`**: Core processing and retrieval logic
- **`Collection`**: Milvus database interface
- **`EnsembleRetriever`**: Hybrid search implementation
- **`CrossEncoder`**: Result reranking

### Extension Points
- **Custom Embeddings**: Replace OpenAI with local models
- **Additional Retrievers**: Add more search strategies
- **Document Types**: Extend beyond PDF support
- **UI Components**: Enhance Streamlit interface

## 🔍 Advanced Features

### Hybrid Search Strategy
1. **Sparse Retrieval (BM25)**: Excellent for keyword matching
2. **Dense Retrieval (Vector)**: Captures semantic similarity
3. **Ensemble Weighting**: Optimized 60/40 split
4. **Cross-encoder Reranking**: Final relevance scoring

### Visual Context
- Automatic image extraction from PDFs
- Context-aware image display
- Support for charts, diagrams, and figures
- Metadata linking to source pages

### PDF Management
- Duplicate detection and prevention
- Per-document statistics tracking
- Bulk upload and processing
- Clean deletion with file cleanup

## 🧪 Examples

### Sample Queries
```python
# Technical questions
"What is multi-head attention?"
"How does the transformer architecture work?"

# Comparative analysis
"Compare BERT and GPT models"
"What are the differences between RNNs and Transformers?"

# Specific details
"What are the hyperparameters used in the model?"
"How many parameters does LLAMA 2 have?"
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📧 Support

For questions or issues:
- Open a GitHub issue
- Check the documentation
- Review the example notebooks

---

**Built with**: Streamlit, Milvus, LangChain, OpenAI, and Unstructured