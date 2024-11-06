import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit App
st.title("PDF Analysis using RAG")
st.write("Upload a PDF document, ask questions, and get answers based on the document content.")

# Upload and load PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    st.success("PDF loaded successfully!")

    # Split document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    st.write(f"Document split into {len(chunks)} chunks.")

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    CHROMA_PATH = "chroma_db"
    db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    st.success("Document embedded and indexed successfully!")

    # Prompt template
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Dont justify your answers.
    Dont give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # User query input
    query = st.text_input("Ask a question about the document:")
    if query:
        st.write(f"Question: {query}")
        
        # Retrieve and generate answer
        docs_chroma = db_chroma.similarity_search_with_score(query, k=5)
        context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

        print("Context text", context_text)
        
        prompt = prompt_template.format(context=context_text, question=query)
        
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        response_text = model.predict(prompt)
        
        st.subheader("Answer")
        st.write(response_text)
