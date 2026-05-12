import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

st.title("InsightFlow")
st.write("An Enterprise Knowledge Intelligence Agent")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    st.subheader("Extracted Text Preview")
    st.text_area("PDF Content", text[:2000], height=250)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    st.success(f"Created {len(chunks)} text chunks.")

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_texts(
        chunks,
        embeddings
    )

    question = st.text_input("Ask a question about the document")

    if question:

        docs = vectorstore.similarity_search(question, k=3)

        st.subheader("Relevant Chunks")

        for i, doc in enumerate(docs):
            st.write(f"Chunk {i+1}")
            st.write(doc.page_content)
            st.divider()
