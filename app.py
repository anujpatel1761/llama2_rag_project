import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import time
from datetime import datetime, timedelta


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
    def estimate_processing_time(self, file_size):
        # Rough estimation based on file size (in MB)
        size_mb = file_size / (1024 * 1024)
        # Assume 1MB takes about 2 seconds to process
        estimated_seconds = size_mb * 2
        return estimated_seconds
        
    def load_document(self, file):
        name, ext = os.path.splitext(file.name)
        file_path = f"temp_{name}{ext}"
        
        # Calculate estimated time
        estimated_time = self.estimate_processing_time(file.size)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        start_time = time.time()
        
        try:
            # Initialize appropriate loader
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                status_text.text("Loading PDF...")
                progress_bar.progress(20)
            elif ext == ".txt":
                loader = TextLoader(file_path)
                status_text.text("Loading text file...")
                progress_bar.progress(20)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
                status_text.text("Loading Word document...")
                progress_bar.progress(20)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Load documents
            documents = loader.load()
            progress_bar.progress(40)
            status_text.text("Splitting text into chunks...")
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            progress_bar.progress(60)
            status_text.text("Processing chunks...")
            
            # Calculate actual processing time
            elapsed_time = time.time() - start_time
            remaining_time = max(0, estimated_time - elapsed_time)
            
            # Update progress
            progress_bar.progress(80)
            status_text.text(f"Almost done... {int(remaining_time)}s remaining")
            
            # Cleanup
            os.remove(file_path)
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            return chunks
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.llm = CTransformers(
            model="./models/llama-2-7b-chat.ggmlv3.q4_K_S.bin",  # Update this path to where you store the model
            model_type="llama",
            max_new_tokens=512,
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
    def create_vectorstore(self, chunks):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Creating embeddings...")
        progress_bar.progress(30)
        
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        progress_bar.progress(100)
        status_text.text("Vector store created successfully!")
        time.sleep(1)  # Show completion message briefly
        status_text.empty()
        progress_bar.empty()
        
        return vectorstore
        
    def create_conversation_chain(self, vectorstore):
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory
        )
        return chain

def main():
    st.set_page_config(page_title="Document Q&A System")
    st.header("Chat with your Documents using Llama 2")
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # File upload section with file size limit
    st.write("Upload your document (Max size: 10MB)")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'docx'])
    
    if uploaded_file:
        # Check file size
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error("File size exceeds 10MB limit. Please upload a smaller file.")
            return
            
        processor = DocumentProcessor()
        rag_system = RAGSystem()
        
        try:
            # Process document with progress tracking
            chunks = processor.load_document(uploaded_file)
            vectorstore = rag_system.create_vectorstore(chunks)
            st.session_state.conversation = rag_system.create_conversation_chain(vectorstore)
            st.success("Document processed successfully! You can now ask questions.")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return
            
    # Chat interface
    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about your document:")
        
        if user_question:
            with st.spinner("Generating response..."):
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history.append((user_question, response['answer']))
                
        # Display chat history
        for question, answer in st.session_state.chat_history:
            with st.container():
                st.write("Question:", question)
                st.write("Answer:", answer)
                st.markdown("---")

if __name__ == "__main__":
    main()