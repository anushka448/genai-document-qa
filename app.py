import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="GenAI Document QA",
    page_icon="🤖",
    layout="centered"
)

st.title("📄 GenAI Document Question Answering")
st.write("Ask questions from your PDF document using Generative AI (FREE & Open Source).")

# Load embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Load FAISS vector database
@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    return FAISS.load_local("faiss_index", embeddings)

# Load QA model
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

db = load_vector_db()
qa_model = load_qa_model()

# User input
query = st.text_input("🔍 Enter your question:")

if query:
    with st.spinner("Thinking... 🤔"):
        docs = db.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in docs])

        result = qa_model(
            question=query,
            context=context
        )

    st.subheader("✅ Answer")
    st.success(result["answer"])
