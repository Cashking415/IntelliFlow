import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

st.set_page_config(
    page_title="InsightFlow",
    page_icon="📄",
    layout="wide"
)

st.title("InsightFlow")
st.caption("Enterprise Knowledge Intelligence Agent")

with st.sidebar:
    st.header("About InsightFlow")
    st.write(
        "InsightFlow helps organizations search, summarize, and analyze internal business documents "
        "using Retrieval Augmented Generation."
    )

    st.subheader("Core Features")
    st.write("• PDF document upload")
    st.write("• Semantic document search")
    st.write("• AI-generated answers")
    st.write("• Source-grounded responses")
    st.write("• Hallucination guardrails")

    st.subheader("Tech Stack")
    st.write("Python · Streamlit · LangChain · ChromaDB · OpenAI")

st.markdown("### Upload Business Document")
uploaded_file = st.file_uploader(
    "Upload a PDF file to begin",
    type="pdf"
)

if uploaded_file:
    with st.spinner("Reading document..."):
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

    st.success("Document uploaded and processed successfully.")

    with st.expander("View Extracted Text Preview"):
        st.text_area("PDF Content", text[:3000], height=250)

    with st.spinner("Creating document knowledge base..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150
        )

        chunks = splitter.split_text(text)

        embeddings = OpenAIEmbeddings()

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings
        )

    st.info(f"Knowledge base ready. Created {len(chunks)} searchable text chunks.")

    st.markdown("### Ask a Question")

    question = st.text_input(
        "Ask something about the uploaded document",
        placeholder="Example: What are the main points in this document?"
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2
    )

    col1, col2 = st.columns(2)

    with col1:
        ask_button = st.button("Generate Answer")

    with col2:
        summary_button = st.button("Generate Executive Summary")

    if ask_button and question:
        with st.spinner("Searching document and generating answer..."):
            docs = vectorstore.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are InsightFlow, an enterprise knowledge assistant.

Use ONLY the provided document context to answer the user's question.

Rules:
- Give a clear, professional answer.
- Use bullet points when helpful.
- Do not make up information.
- If the answer is not found in the context, say:
  "I could not find that information in the uploaded document."
- Keep the answer concise but useful.

Document Context:
{context}

User Question:
{question}
"""

            try:
                response = llm.invoke(prompt)

                st.markdown("### AI Answer")
                st.write(response.content)

                st.markdown("### Source Evidence")
                for i, doc in enumerate(docs):
                    with st.expander(f"Source Chunk {i + 1}"):
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"OpenAI Error: {e}")

    if summary_button:
        with st.spinner("Generating executive summary..."):
            docs = vectorstore.similarity_search(
                "main points key ideas summary overview conclusions",
                k=6
            )

            context = "\n\n".join([doc.page_content for doc in docs])

            summary_prompt = f"""
You are InsightFlow, an enterprise knowledge assistant.

Create an executive summary of the uploaded document using ONLY the context below.

Format the summary with:
1. Overview
2. Key Points
3. Business Relevance
4. Important Takeaways

If the context is not enough, say what information is missing.

Document Context:
{context}
"""

            try:
                summary_response = llm.invoke(summary_prompt)

                st.markdown("### Executive Summary")
                st.write(summary_response.content)

            except Exception as e:
                st.error(f"OpenAI Error: {e}")

else:
    st.warning("Upload a PDF document to begin.")
