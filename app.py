import os
import streamlit as st
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
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
    # 🔍 Q&A
if question:

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(model="gpt-4o-mini")

    # 🔥 NEW ADAPTIVE PROMPT
    prompt = f"""
You are a smart AI assistant helping a student.

Use ONLY the provided context to answer the question.

IMPORTANT RULES:
- Follow exactly what the user is asking
- Adapt your answer style based on the question:
    • If user asks for short answer → keep it concise
    • If user asks for explanation → explain clearly
    • If user asks for bullet points → use bullet points
    • If user asks for summary → summarize
    • If user asks for comparison → structure it properly
- Do NOT force any format unless the user asks
- Keep language simple and natural

Context:
{context}

User Question:
{question}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.markdown(response.content)  
