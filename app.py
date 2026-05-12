import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

st.title("InsightFlow")
st.write("An Enterprise Knowledge Intelligence Agent")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # Display extracted text
    st.subheader("Extracted Text Preview")
    st.text_area("PDF Content", text[:2000], height=250)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    st.success(f"Created {len(chunks)} text chunks.")

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Store chunks in ChromaDB
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    # User question
    question = st.text_input("Ask a question about the document")

    if question:

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(question, k=3)

        # Combine retrieved chunks
        context = "\n\n".join([doc.page_content for doc in docs])

        # OpenAI chat model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )

        # Prompt
        prompt = f"""
You are an enterprise knowledge assistant.

Use ONLY the provided context to answer the question.

If the answer is not found in the context, say:
"I could not find that information in the uploaded document."

Context:
{context}

Question:
{question}
"""

        # Generate response
        response = llm.invoke(prompt)

        # Display AI answer
        st.subheader("AI Answer")
        st.write(response.content)

        # Display retrieved chunks
        st.subheader("Retrieved Sources")

        for i, doc in enumerate(docs):
            st.write(f"Source Chunk {i + 1}")
            st.write(doc.page_content)
            st.divider()
