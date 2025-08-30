# Smart Academic Assistant

AI-powered academic assistant using RAG (Retrieval-Augmented Generation). Upload study materials in PDF format and get instant answers to your questions using Google Gemini, LangChain, HuggingFace embeddings, and ChromaDB.

## Features

- Upload PDF documents (notes, textbooks, study materials)
- Automatically create vector-based knowledge base
- Ask natural language questions about content
- Context-aware answers using Gemini model
- Simple Streamlit web interface

## Tech Stack

- **Python** - Core development
- **Streamlit** - Web interface
- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **HuggingFace Sentence Transformers** - Embeddings
- **Google Generative AI Gemini** - Language model

## Installation

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set Google credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

### Run Application
```bash
streamlit run app.py
```

## Authentication

Uses Google JSON credentials file instead of API key. Set the environment variable:

```env
GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/your/credentials.json"
```

## Use Cases

- Extract specific information from student lists (birthdays, phone numbers, exam dates)
- Query uploaded lecture notes with specific questions
- Review PDF study guides through summary questions
- Search through academic documents and textbooks

## Limitations

- Works best with well-formatted text PDFs
- Raw unstructured data may require proper formatting
- Document chunking might split relevant information
- Performance depends on PDF text quality

## License

MIT License
