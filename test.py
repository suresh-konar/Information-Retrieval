#Import Libraries
from langchain.document_loaders import TextLoader
import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

import streamlit as st
from openai import OpenAI

# Function to collect URLs from user input
def get_urls():
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)
    return urls

# Function to trigger processing
def process_urls(urls):
    # Placeholder for main content
    main_placeholder = st.empty()

    # Initialize OpenAI model
    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.9, max_tokens=500)
    
    # Placeholder for processing results
    with main_placeholder:
        if urls:
            for url in urls:
                # Process each URL
                # You can add your processing logic here
                st.write(f"Processing URL: {url}")
        else:
            st.write("Please enter URLs on the sidebar and press Enter.")

# Set page title and sidebar title
st.title("InfoRetrieveX")
st.sidebar.title("Paste URL's below:")

# Collect URLs from user input
urls = get_urls()

# Button to trigger processing
process_url_clicked = st.sidebar.button("Press Enter")

# Define file path for storing Faiss embeddings
file_path = "faiss_store_openai.pkl"

# Initialize OpenAI model
llm = OpenAI( model='gpt-3.5-turbo-instruct', temperature=0.9, max_tokens=500)


if process_url_clicked:
    # Initialize a loader object to load data from the provided URLs
    loader = UnstructuredURLLoader(urls=urls)
    # Load data from the specified URLs using the loader object
    data = loader.load()

    # Initialize a text splitter object with defined separators and chunk size for splitting the text into smaller segments
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    # Split the loaded data into smaller documents using the text splitter object
    docs = text_splitter.split_documents(data)

    # Initialize a placeholder for status updates
    status_placeholder = st.empty()

    # Function to update status
    def update_status(message):
        status_placeholder.text(message)

    # Update the main content placeholder to indicate that data loading has started
    update_status("Data Loading...Started...✅✅✅")

    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # Update the status
    update_status("Embedding Vector Building...Started...✅✅✅")

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Prompt the user to input a question via the main content placeholder
query = st.text_input("Question: ")


if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)
            else:
                st.write("No sources found for this question.")
    else:
        st.write("FAISS index file not found. Please load data and build embeddings first.")
else:
    st.write("Please enter a question to retrieve information.")
