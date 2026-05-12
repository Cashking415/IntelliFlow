import streamlit as st
from pypdf import PdfReader

st.title("InsightFlow")
st.write("An Enterprise Knowledge Intelligence Agent")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    st.subheader("Extracted Text Preview")
    st.text_area("PDF Content", text[:3000], height=300)
