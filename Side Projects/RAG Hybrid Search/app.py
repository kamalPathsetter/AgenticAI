import os
# Fix for tokenizers warning - must be set before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
from rag_pipeline import RAGPipeline
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    try:
        st.session_state.rag_pipeline = RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

# App title and description
st.title("📚 RAG PDF Assistant")
st.markdown("Upload PDFs, ask questions, and get intelligent answers with **hybrid search** and visual context!")

# Add info about hybrid search
with st.expander("ℹ️ About Hybrid Search"):
    st.markdown("""
    This RAG system uses **hybrid search** combining:
    - **BM25 (Sparse Retrieval)**: Excellent for keyword matching and exact terms
    - **Vector Search (Dense Retrieval)**: Great for semantic similarity and context understanding
    - **Cross-encoder Reranking**: Final reranking for optimal relevance
    
    The ensemble approach provides more comprehensive and accurate results!
    """)

# Sidebar for PDF management
with st.sidebar:
    st.header("📁 PDF Management")
    
    # Display collection statistics
    stats = st.session_state.rag_pipeline.get_collection_stats()
    st.metric("Total PDFs", stats["total_pdfs"])
    st.metric("Total Chunks", stats["total_chunks"])
    
    st.divider()
    
    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to your knowledge base"
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if already processed
            if not st.session_state.rag_pipeline.check_pdf_exists(uploaded_file.name):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Process PDF
                        documents = st.session_state.rag_pipeline.process_pdf(
                            tmp_file_path, 
                            uploaded_file.name
                        )
                        
                        if documents:
                            # Insert into database
                            if st.session_state.rag_pipeline.insert_documents(documents):
                                st.success(f"✅ Processed {uploaded_file.name} ({len(documents)} chunks)")
                                st.rerun()  # Refresh to update the PDF list
                            else:
                                st.error(f"❌ Failed to store {uploaded_file.name}")
                        else:
                            st.warning(f"⚠️ No content extracted from {uploaded_file.name}")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
            else:
                st.info(f"📄 {uploaded_file.name} already exists in database")
    
    st.divider()
    
    # Display stored PDFs with enhanced information
    st.subheader("📋 Stored PDFs")
    stored_pdfs = st.session_state.rag_pipeline.get_stored_pdfs()
    
    if stored_pdfs:
        for pdf_info in stored_pdfs:
            with st.expander(f"📄 {pdf_info['name']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chunks", pdf_info['chunks'])
                with col2:
                    st.metric("Pages", pdf_info['pages'])
                with col3:
                    st.metric("Images", pdf_info['images'])
                
                # Delete button
                if st.button(
                    "🗑️ Delete PDF", 
                    key=f"delete_{pdf_info['name']}", 
                    help=f"Delete all data for {pdf_info['name']}",
                    type="secondary"
                ):
                    with st.spinner(f"Deleting {pdf_info['name']}..."):
                        if st.session_state.rag_pipeline.delete_pdf_records(pdf_info['name']):
                            st.success(f"✅ Deleted {pdf_info['name']}")
                            st.rerun()  # Refresh the page
                        else:
                            st.error(f"❌ Failed to delete {pdf_info['name']}")
    else:
        st.info("No PDFs stored yet. Upload some PDFs to get started!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is multi-head attention?",
        help="Ask questions about the content in your uploaded PDFs"
    )
    
    # Search configuration
    with st.expander("🔧 Hybrid Search Settings"):
        col_a, col_b = st.columns(2)
        with col_a:
            num_results = st.slider("Documents to retrieve", 1, 10, 5, help="More documents = broader search scope")
            num_docs_for_answer = st.slider("Documents for answer", 1, 5, 2, help="Documents used to generate the final answer")
        
        with col_b:
            # PDF filter option
            pdf_options = ["All PDFs"] + [pdf["name"] for pdf in stored_pdfs]
            selected_pdf = st.selectbox(
                "Search in specific PDF:",
                pdf_options,
                help="Choose a specific PDF to search in, or select 'All PDFs'"
            )
            pdf_filter = None if selected_pdf == "All PDFs" else selected_pdf
            
            # Show search method info
            st.info("🔄 **Hybrid Search**: BM25 (60%) + Vector (40%) + Cross-encoder reranking")
    
    # Search and generate answer
    if st.button("🔍 Hybrid Search & Answer", type="primary") and query:
        if not stored_pdfs:
            st.warning("⚠️ Please upload some PDFs first!")
        else:
            with st.spinner("🔄 Running hybrid search and generating answer..."):
                # Search and rank documents using hybrid approach
                ranked_docs = st.session_state.rag_pipeline.search_and_rank(
                    query, num_results, pdf_filter
                )
                
                if ranked_docs:
                    # Generate answer
                    answer, top_docs = st.session_state.rag_pipeline.generate_answer(
                        query, ranked_docs, num_docs_for_answer
                    )
                    
                    # Display answer
                    st.subheader("🎯 Answer")
                    st.write(answer)
                    
                    # Display search methodology used
                    search_method = "🔄 **Hybrid Search Used**: BM25 + Vector Search + Cross-encoder Reranking"
                    if pdf_filter:
                        search_method += f" (filtered to: {pdf_filter})"
                    st.caption(search_method)
                    
                    # Display source information
                    st.subheader("📄 Sources")
                    for i, doc in enumerate(top_docs, 1):
                        with st.expander(f"Source {i} - {doc.metadata.get('pdf_name', 'Unknown')} (Page {doc.metadata.get('page_number', 'Unknown')})"):
                            if doc.metadata.get('section_title'):
                                st.write(f"**Section:** {doc.metadata.get('section_title')}")
                            st.write(f"**Content Preview:**")
                            preview_text = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.write(preview_text)
                    
                    # Store results in session state for image display
                    st.session_state.current_docs = top_docs
                    
                    # Show retrieval stats
                    with st.expander("📊 Retrieval Statistics"):
                        st.write(f"- **Total documents retrieved**: {len(ranked_docs)}")
                        st.write(f"- **Documents used for answer**: {len(top_docs)}")
                        st.write(f"- **Search scope**: {'All PDFs' if not pdf_filter else pdf_filter}")
                        st.write(f"- **Retrieval method**: Hybrid (BM25 + Vector + Reranking)")
                else:
                    search_scope = f" in {pdf_filter}" if pdf_filter else ""
                    st.warning(f"⚠️ No relevant documents found for your query{search_scope}.")

with col2:
   st.header("🖼️ Visual Context")
   
   # Display images from current search results
   if hasattr(st.session_state, 'current_docs') and st.session_state.current_docs:
       image_found = False
       
       for i, doc in enumerate(st.session_state.current_docs, 1):
           image_paths = doc.metadata.get("image_path", "")
           if image_paths:
               # Handle both string and list formats
               if isinstance(image_paths, str):
                   paths = [p.strip() for p in image_paths.split(",") if p.strip()]
               else:
                   paths = image_paths
               
               for j, path in enumerate(paths):
                   if os.path.exists(path):
                       try:
                           img = Image.open(path)
                           caption = f"{doc.metadata.get('pdf_name', 'Unknown')} - Page {doc.metadata.get('page_number', '?')} - Image {j+1}"
                           st.image(
                               img, 
                               caption=caption,
                               use_container_width=True
                           )
                           image_found = True
                       except Exception as e:
                           st.error(f"Error loading image: {e}")
       
       if not image_found:
           st.info("No images found in the current search results.")
   else:
       st.info("Perform a hybrid search to see related images here.")

# Footer
st.markdown("---")
st.markdown(
   """
   💡 **Enhanced RAG Features:**
   - 🔄 **Hybrid Search**: Combines BM25 and vector search for optimal results
   - 🧠 **Cross-encoder Reranking**: Improves relevance scoring
   - 📊 **PDF Management**: Track and manage multiple documents
   - 🖼️ **Visual Context**: Automatically extract and display relevant images
   - ⚙️ **Configurable**: Adjust search parameters for your needs
   """
)