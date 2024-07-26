# Information Retrieval - Research Tool

# Overview
 I propose developing a user-friendly research tool aimed at facilitating effortless
 information retrieval. The tool will allow users to input article URLs or upload text files
 containing URLs to fetch article content. This content will be processed through
 LangChain's Unstructured URL Loader to extract pertinent information. Leveraging
 OpenAI's embeddings, I will construct embedding vectors for the articles, and utilize FAISS,
 a powerful similarity search library, to enable swift and effective retrieval of relevant
 information. Additionally, users will be able to interact with the Language Model (LLM),
 powered by ChatGPT, by inputting queries and receiving answers along with source URLs.

# Features
 - URLInput:Userscaninput article URLs directly into the tool interface or upload text files containing URLs for batch processing.
 - ContentProcessing: The tool will utilize LangChain's Unstructured URL Loader to process the content of the articles fetched from the provided URLs.
 - EmbeddingConstruction: We will employ OpenAI's embeddings to construct embedding vectors for the articles, facilitating efficient information retrieval.
 - Similarity Search: FAISS will be integrated into the tool to perform similarity search on the embedding vectors, enabling quick and effective retrieval of relevant information based on user queries.
 - Interactivity with ChatGPT: Users can interact with ChatGPT, powered by LLM, by inputting queries related to the financial domain, and receive answers along with source URLs for further exploration  
