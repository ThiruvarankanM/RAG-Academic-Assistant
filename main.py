import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

# Load environment variables
load_dotenv(override=True)

# Constants
DB_PATH = './vector_store'
PDF_PATH = './uploaded_pdfs'
COLLECTION_NAME = "academic_qa_bot"

# Streamlit App UI
st.title("Smart Academic Assistant")
st.caption("Built with RAG and Gemini")

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(PDF_PATH, exist_ok=True)

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Session State for History
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# File Upload
st.subheader("Upload Study Materials")
uploaded_files = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(PDF_PATH, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success("Files uploaded successfully.")

# Embed and Store
if st.button("Generate Knowledge Base"):
    with st.spinner("Processing documents..."):
        all_docs = []
        for filename in os.listdir(PDF_PATH):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_PATH, filename))
                docs = loader.load()
                for page in docs:
                    page.metadata["source"] = filename
                all_docs.extend(docs)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(all_docs)

        # Add UUID to each chunk metadata
        for chunk in chunks:
            chunk.metadata["id"] = str(uuid4())

        # Save to Chroma DB (overwrite existing DB)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME
        )
        # NO .persist() here because it causes error

    st.success(f"{len(chunks)} document chunks indexed successfully.")

# Query Section
st.subheader("Ask a Question")
user_query = st.text_input("Enter your question")

if user_query:
    with st.spinner("Retrieving answer..."):
        try:
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=DB_PATH
            )

            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            related_docs = retriever.get_relevant_documents(user_query)

            if related_docs:
                st.markdown("### Top Matching Passages")
                for i, doc in enumerate(related_docs, 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Passage {i}:** (Page {page}, File: {source})")
                    st.write(doc.page_content.strip())

                system_prompt = (
                    "You are an academic assistant. Answer questions based only on the given context. "
                    "Be brief and informative. If you don't know the answer, say you don't know.\n\n{context}"
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])

                qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
                rag_chain = create_retrieval_chain(retriever, qa_chain)

                result = rag_chain.invoke({"input": user_query})
                final_answer = result["answer"]

                st.subheader("Answer")
                st.markdown(final_answer)

                st.session_state.qa_history.append((user_query, final_answer))
            else:
                st.warning("No relevant content found.")

        except Exception as e:
            st.error(f"Error: {e}")

# History Viewer
if st.session_state.qa_history:
    st.subheader("Previous Questions")
    for q, a in reversed(st.session_state.qa_history):
        st.markdown(f"**Question:** {q}")
        st.markdown(f"**Answer:** {a}")
        st.markdown("---")
