# Import necessary libraries
import os                               # Provides functions for interacting with the operating system
from dotenv import load_dotenv          # Loads environment variables from a .env file
import streamlit as st                  # Streamlit is used to create the web app interface

# Import specific classes from the LangChain library
from langchain_openai import ChatOpenAI                              # Chat interface for OpenAI's language models
from langchain_openai import OpenAIEmbeddings                        # Generates embeddings using OpenAI's API
from langchain.text_splitter import RecursiveCharacterTextSplitter   # Splits text into chunks
from langchain_community.vectorstores import FAISS                   # Using FAISS as the vector store
from langchain.chains import ConversationalRetrievalChain            # Handles retrieval-based question-answering


# Load .env file
load_dotenv()
open_ai_api_key = os.getenv("OPEN_AI_KEY")


def clear_history():
    """Clears the chat history in streamlit when a new file is uploaded"""
    if 'history' in st.session_state:
        del st.session_state['history']


st.title('Chat With Your Document')
st.write("Instructions:")
st.write("1. Upload your document")
st.write("2. Click Add File")
st.write("3. Ask questions about your document!")

uploaded_file = st.file_uploader('Upload file:', type=['pdf', 'docx', 'txt'])
add_file = st.button('Add File', on_click=clear_history)

if uploaded_file and add_file:
    with st.spinner('Reading, Chunking, and Embedding File...'):
        # Puts the uploaded file locally onto our drive
        bytes_data = uploaded_file.read()
        file_name = os.path.join('./', uploaded_file.name)

        with open(file_name, 'wb') as f:
            f.write(bytes_data)

        # Splits the file name and extension type, then loads the file based on the extension type
        name, extension = os.path.splitext(file_name)

        if extension == '.pdf':
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_name)
        elif extension == '.docx':
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_name)
        elif extension == '.txt':
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_name)
        else:
            st.write('Document format is not supported')

        documents = loader.load()                     # Loads the content of the file into the documents variable

        # Create a text splitter object, with a custom chunk_size and overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        chunks = text_splitter.split_documents(documents)  # Splits and stores the chunks in chunks

        embeddings = OpenAIEmbeddings(api_key=open_ai_api_key)      # Sets up embeddings using OpenAI

        # Create a FAISS vector store from the documents and embeddings
        vector_store = FAISS.from_documents(chunks, embeddings)    # Create FAISS index for the document chunks

        # We initialize our llm, create a retriever from our FAISS index
        llm = ChatOpenAI(api_key=open_ai_api_key, model='gpt-4-turbo', temperature=0, max_tokens=1000)
        retriever = vector_store.as_retriever()                         # Turns the FAISS store into a retriever
        crc = ConversationalRetrievalChain.from_llm(llm, retriever)     # Our retrieve information becomes part of the llm info it draws from
        st.session_state.crc = crc                                      # Needed so that the session state knows it later

        st.success('File Uploaded, chunked and embedded successfully')

question = st.text_input('Input your question about the document')

if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        response = crc.invoke({
            'question': question,
            'chat_history': st.session_state['history']
        })

        st.session_state['history'].append((question, response['answer']))
        st.write(response['answer'])
