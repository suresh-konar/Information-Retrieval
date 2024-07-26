#Load the required libraries
import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# take environment variables from .env (especially openai api key)
load_dotenv()  

# Set page title and sidebar title
st.title("InfoRetrieveX")
st.sidebar.title("Paste URL's below:")

# Collect URLs from user input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to trigger processing
process_url_clicked = st.sidebar.button("Press Enter")

# Define file path for storing Faiss embeddings
file_path = "faiss_store_openai.pkl"

# Placeholder for main content
main_placeholder = st.empty()

# Initialize OpenAI model
llm = OpenAI( model='gpt-3.5-turbo-instruct', temperature=0.9, max_tokens=500)

#Check if the "Process URLs" button has been clicked
if process_url_clicked:
    
    #Initialize a loader object to load data from the provided URLs
    loader = UnstructuredURLLoader(urls=urls)

    #Update the main content placeholder to indicate that data loading has started
    main_placeholder.text("Data Loading...Started...✅✅✅")

    #Load data from the specified URLs using the loader object
    data = loader.load()

    #Initialize a text splitter object with defined separators and chunk size for splitting the text into smaller segments
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    #Update the main content placeholder to indicate that text splitting has started
    main_placeholder.text("Text Splitter...Started...✅✅✅")

    #Split the loaded data into smaller documents using the text splitter object
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    #Update the main content placeholder to indicate that embedding vector building has started 
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    #and pause execution for 3 seconds
    time.sleep(3)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

#Prompt the user to input a question via the main content placeholder
query = main_placeholder.text_input("Question: ")


#Check if a question has been entered
if query:

    #Check if the file path for the FAISS index exists
    if os.path.exists(file_path):

        #Load the FAISS index from the pickle file
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            #Create a chain object for question answering and source retrieval
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            #Use the chain to process the user's question and retrieve the answer, returning only the outputs
            result = chain({"question": query}, return_only_outputs=True)

            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")

            #Check if sources are available
            if sources:

                #Display a subheader indicating sources 
                st.subheader("Sources:")

                # Split the sources by newline
                sources_list = sources.split("\n")  

                #Iterate through the list of sources and display each one.
                for source in sources_list:
                    st.write(source)