# 📚 Smart Academic Assistant

An AI-powered academic assistant using **RAG (Retrieval-Augmented Generation)**. Upload study materials in PDF format and get instant answers to your questions — powered by **Google Gemini**, **LangChain**, **HuggingFace embeddings**, and **ChromaDB**.

---

## 🚀 Features

* 📄 Upload PDF documents (e.g., notes, textbooks, lists)
* 🧠 Automatically create a vector-based knowledge base
* 🔍 Ask natural language questions about the content
* 🤖 Gemini model answers with context-aware reasoning
* ⚡ Built with **Streamlit** for a simple and fast UI

---

## 📦 Tech Stack

* **Python**
* **Streamlit** (Frontend)
* **LangChain** (RAG framework)
* **ChromaDB** (Vector database)
* **HuggingFace Sentence Transformers** (Embeddings)
* **Google Generative AI Gemini** (LLM API via JSON credentials)

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-academic-assistant.git
cd smart-academic-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> 🔐 **Authentication Note:**
> Instead of using an API key (`GOOGLE_API_KEY`), we use a **Google JSON credentials file** to access the Gemini model.

Make sure to set the following environment variable in your terminal or `.env` file:

```env
GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/your/credentials.json"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## ✅ Example Use Cases

* Ask for **birthdays**, **phone numbers**, or **exam dates** from student lists
* Upload **lecture notes** and ask specific questions
* Review **PDF study guides** by asking summary questions

---

## 📌 Limitations

* Works best with **well-formatted text PDFs**
* Raw unstructured data (like names and numbers in one line) may confuse the model unless formatted properly
* Gemini prefers **clear context**, and chunking might split relevant info unless handled carefully

---

## 📄 License

MIT License

---

## 🤝 Contributions

Feel free to fork and improve! PRs are welcome.

---
