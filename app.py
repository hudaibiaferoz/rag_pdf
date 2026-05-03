import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

st.set_page_config(page_title="AskMyPDF", layout="wide")

st.title("Stop getting khwar")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Ask a question about the PDF")

# Initialize embeddings ONCE
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved: {uploaded_file.name}")

    # 👉 CHECK: if DB already exists
    if not os.listdir("chroma_db"):

        st.info("Processing PDF... (first time only)")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        st.success("Vector DB created!")

    else:
        # 👉 Load existing DB (no reprocessing)
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        st.info("Loaded existing vector DB")

    # 🔍 Q&A
    if question:

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model="gpt-4o-mini")

        prompt = f"""
You are a helpful and engaging tutor.

Your job is to explain answers in a way that is:
- easy to understand
- well-structured
- interesting to read
- suitable for students

Use ONLY the provided context to answer.

Format your answer like this:
1. Start with a short, clear summary
2. Then explain in simple bullet points
3. Use examples if helpful
4. Avoid robotic or overly technical language

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        # Sources
        st.subheader("Sources")
        for i, doc in enumerate(docs):
            st.write(f"Source {i+1}")
            clean_text = doc.page_content.replace("\n", " ").strip()

            st.markdown(f"**Source {i+1}:**")
            st.markdown(clean_text[:300] + "...")
            st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")
            st.divider()    