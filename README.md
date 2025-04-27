# Simple RAG in Streamlit using OpenAI

This project demonstrates a simple Retrieval-Augmented Generation (RAG) application built with Streamlit and OpenAI. The application allows users to upload a PDF document, which is then processed and stored in a vector database. Users can ask questions about the document, and the application retrieves relevant information to generate answers.

## Project Overview

The application covers:

- **Document Upload and Processing**: Allows users to upload PDF files, which are then processed, chunked, and embedded for conversational retrieval.
- **Chat Interface**: Enables users to ask questions about the uploaded document and receive answers based on the document's content.
- **Conversational Retrieval**: Utilizes LangChain's ConversationalRetrievalChain to handle retrieval-based question-answering.

## Project Setup

There are two primary components for this project:

- **tech_demo.py**: The main script that sets up the Streamlit application and handles the document upload, processing, and chat interface.
- **.env**: A file containing the OpenAI API key, which is required for generating embeddings and using the language models.

## Features

### *Document Upload and Processing*

- **Supported File Types**: The application supports PDF files, DOCX, and TXT files.
- **File Processing**: Uploaded files are read, chunked, and embedded using OpenAI's embeddings. The chunks are stored in a FAISS vector store for efficient retrieval.

### *Chat Interface*

- **User Instructions**: The application provides clear instructions for users to upload their documents and ask questions.
- **Question Input**: Users can input their questions about the document through a text input field.

### *Conversational Retrieval*

- **Retrieval-Based Question-Answering**: The application uses LangChain's ConversationalRetrievalChain to retrieve information from the document based on the user's questions.
- **Chat History**: The application maintains a chat history to provide context for follow-up questions.

## Acknowledgements

- [Streamlit](https://streamlit.io/) - For providing the framework to create the web app interface.
- [LangChain](https://github.com/hwchase17/langchain) - For providing the tools for conversational retrieval and document processing.
- [OpenAI](https://openai.com/) - For providing the language models and embeddings used in the application.
