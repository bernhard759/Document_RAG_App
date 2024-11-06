import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
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
st.title("Enhanced PDF Analysis using RAG")
st.write("Upload PDF documents, ask questions, and get answers based on the document content.")

# Adjustable Parameters
k = st.slider("Number of chunks to retrieve", 1, 10, 5)
chunk_overlap = st.slider("Chunk overlap", 0, 100, 50)
CHROMA_PATH = "chroma_db"  # Directory to store Chroma database

# Cache function for embedding and indexing
@st.cache_resource
def load_and_index_documents(_chunks, _embeddings, path):
    db_chroma = Chroma.from_documents(_chunks, _embeddings, persist_directory=path)
    return db_chroma


# File upload with multi-file support
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
all_chunks = []

if uploaded_files:
    for file in uploaded_files:
        with open(f"temp_{file.name}", "wb") as f:
            f.write(file.getbuffer())
        
        # Use PyPDFLoader to load and process each PDF
        loader = PyPDFLoader(f"temp_{file.name}")
        pages = loader.load()

        # Split document into smaller chunks with adjustable overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
    
    st.success(f"{len(uploaded_files)} PDFs loaded and split into {len(all_chunks)} chunks.")

    # Initialize embeddings and create the vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db_chroma = load_and_index_documents(all_chunks, embeddings, CHROMA_PATH)
    st.success("All documents embedded and indexed successfully!")

    # Model
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # Option to generate a summary of the document
    if st.button("Summarize Document"):
        full_text = "\n\n".join([chunk.page_content for chunk in all_chunks])
        summary_prompt = f"Summarize the following document:\n\n{full_text}"
        summary = model.predict(summary_prompt)
        st.subheader("Summary")
        st.write(summary)

    # Define the prompt template
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

    # Initialize chat history for conversational follow-up questions
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User query input for asking questions
    query = st.text_input("Ask a question about the document:")
    if query:
        # Add user query to chat history
        st.session_state.chat_history.append(f"User: {query}")

        # Retrieve relevant document chunks based on user query
        docs_chroma = db_chroma.similarity_search_with_score(query, k=k)
        context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

        # Build the prompt with chat history and retrieved context
        full_prompt = "\n".join(st.session_state.chat_history) + "\n\nContext:\n" + context_text + "\nAnswer:"
        
        # Generate answer based on the prompt
        response_text = model.predict(full_prompt)
        
        # Add AI response to chat history
        st.session_state.chat_history.append(f"AI: {response_text}")

        # Display answer with improved formatting
        st.subheader("Answer")
        st.markdown(response_text)
