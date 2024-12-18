# Llama 2 Document Q&A System

A powerful document question-answering system built with Llama 2, Streamlit, and LangChain. This application allows users to upload documents and ask questions about their content, leveraging the power of RAG (Retrieval-Augmented Generation) for accurate responses.

## 🌟 Features

- Document processing support for multiple formats (PDF, TXT, DOCX)
- Real-time progress tracking with estimated completion times
- Efficient text chunking and embedding generation
- Conversational memory for context-aware responses
- User-friendly Streamlit interface
- FAISS vector store for efficient similarity search
- File size limit handling (10MB max)

## 🛠️ Technical Architecture

- **Frontend**: Streamlit
- **Language Model**: Llama 2 (7B-chat quantized)
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: LangChain document loaders
- **Memory**: ConversationBufferMemory

## 📋 Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)
- Llama 2 model file (7B-chat quantized version)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/anujpatel1761/llama2_rag_project.git
cd llama2_rag_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the Llama 2 model:
- Download the quantized model file (llama-2-7b-chat.ggmlv3.q4_K_S.bin)
- Place it in the `models` directory

## 📦 Required Packages

```
streamlit
langchain
langchain_community
langchain_huggingface
ctransformers
sentence-transformers
faiss-cpu
pypdf
python-docx
docx2txt
```

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser (typically http://localhost:8501)

3. Upload your document (PDF, TXT, or DOCX format)

4. Wait for the processing to complete

5. Start asking questions about your document!

## 💡 How It Works

1. **Document Processing**:
   - User uploads a document
   - System validates file type and size
   - Document is loaded and split into chunks
   - Progress is tracked and displayed

2. **Embedding Creation**:
   - Text chunks are converted to embeddings
   - Embeddings are stored in FAISS vector store

3. **Question Answering**:
   - User asks a question
   - System retrieves relevant chunks
   - Llama 2 generates contextual response
   - Chat history is maintained

## ⚠️ Limitations

- Maximum file size: 10MB
- Supported file formats: PDF, TXT, DOCX
- RAM usage depends on document size
- Processing time varies with document length

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- Llama 2 team at Meta AI
- Streamlit team
- LangChain community

## 📧 Contact

Anuj Patel - patel.anuj2@northeastern.edu

Project Link: https://github.com/anujpatel1761/llama2_rag_project
