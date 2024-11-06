# Document RAG App

This project is a **Retrieval-Augmented Generation (RAG) Web Application** for document analysis. Using this app, users can upload PDF files, ask questions about the content, and receive context-aware answers.

## Features

- **PDF Upload and Indexing**: Upload single or multiple PDF documents.
- **Chunk-Based Document Retrieval**: Splits document content into chunks for efficient retrieval.
- **Contextual Q&A**: Provides answers based on the retrieved document context.
- **Document Summarization**: Summarizes documents to provide users with a quick overview.
- **Adjustable Retrieval Parameters**: Customize retrieval chunk count and overlap for granular control.

## Installation

1. Clone the repository and `cd` into it

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

1. Run the app:
   ```bash
   streamlit run app.py
   ```

2. Upload PDFs, select a chat model, and ask questions about the document content.

## References

For a deeper understanding of Retrieval-Augmented Generation (RAG) and building RAG applications, see these articles:

- [Build Your RAG Web Application with Streamlit](https://medium.com/@alb.formaggio/build-your-rag-web-application-with-streamlit-7673120a9741) by Alberto Formaggio
- [What is Retrieval-Augmented Generation (RAG)?](https://medium.com/@drjulija/what-is-retrieval-augmented-generation-rag-938e4f6e03d1) by Dr. Julija
