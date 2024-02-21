import streamlit as st
import pandas as pd
from llama_index import SimpleDirectoryReader, ServiceContext, StorageContext, VectorStoreIndex, SQLDatabase
from llama_index.llms import OpenAI
from llama_index.embeddings import FastEmbedEmbedding
from qdrant_client import QdrantClient
import json
import os
from sqlalchemy import create_engine
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from pathlib import Path
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.query_engine import (
    SQLAutoVectorQueryEngine,
    RetrieverQueryEngine
)

from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.vector_store import VectorIndexAutoRetriever

from llama_index.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever
)

from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.query_engine.retriever_query_engine import (
    RetrieverQueryEngine
)

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout="wide")
write_dir = Path("textdata")

client = QdrantClient(
    url=os.environ['QDRANT_URL'], 
    api_key=os.environ['QDRANT_API_KEY'],
)

def get_text_data(data):
    return f"""   
    Rank: {data['Rank']}
    Movie_name: {data['Movie_name']}
    Year: {data['Year']}
    Certificate: {data['Certificate']}
    Runtime_in_min: {data['Runtime_in_min']}
    Genre: {data['Genre']}
    Rating_from_10: {data['Rating_from_10']}
    """

# Streamlit UI setup
st.title('Movie Recommendation System')

# Load the dataset
df_file_path = 'imdb.csv'  # Path to the csv file
if os.path.exists(df_file_path):
    df = pd.read_csv(df_file_path)
    df["text"] = df.apply(get_text_data, axis=1)
    st.dataframe(df)  # Display df in the UI
else:
    st.error("Data file not found. Please check the path and ensure it's correct.")


def create_text_and_embeddings():
    # Write text data to 'textdata' folder and creating individual files
    if write_dir.exists():
        print(f"Directory exists: {write_dir}")
        [f.unlink() for f in write_dir.iterdir()]
    else:
        print(f"Creating directory: {write_dir}")
        write_dir.mkdir(exist_ok=True, parents=True)

    for index, row in df.iterrows():
        if "text" in row:
            file_path = write_dir / f"Movie{index}.txt"
            with file_path.open("w", encoding='utf-8') as f:
                f.write(str(row["text"]))
        else:
            print(f"No 'text' column found at index {index}")

    print(f"Files created in {write_dir}")
# create_text_and_embeddings()   #execute only once in the beginning

@st.cache_data
def load_data():
    if write_dir.exists():
        reader = SimpleDirectoryReader(input_dir="textdata")
        documents = reader.load_data()
    return documents

documents = load_data()

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(chunk_size_limit=1024, llm=llm, embed_model=embed_model)

vector_store = QdrantVectorStore(client=client, collection_name="movies")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Create vector indexes and store in Qdrant. To be run only once in the beginning
from llama_index import VectorStoreIndex
# index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, service_context=service_context, storage_context=storage_context)

# Load the vector index from Qdrant collection
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)


# Input from user
user_query = st.text_input("Describe the type of movie you want to watch today:", "I want to watch a trending sci-fi movie.")

if st.button("Submit Query"):
        # Part 1, semantic search + LLM call
        # Generate query vector
        query_vector = embed_model.get_query_embedding(user_query)
        # Perform search with Qdrant
        response = client.search(collection_name="movies", query_vector=query_vector, limit=10)
        # Processing and displaying the results
        text = ''
        movie_list = []  # List to store multiple property dictionaries
        for scored_point in response:
            # Access the payload, then parse the '_node_content' JSON string to get the 'text'
            node_content = json.loads(scored_point.payload['_node_content'])
            text += f"\n{node_content['text']}\n"    
            # Initialize a new dictionary for the current property
            movie_dict = {}
            for line in node_content['text'].split('\n'):
                if line.strip():  # Ensure line is not empty
                    key, value = line.split(': ', 1)
                    movie_dict[key.strip()] = value.strip()
            # Add the current property dictionary to the list
            movie_list.append(movie_dict)

        # movie_list contains all the retrieved property dictionaries
        with st.status("Retrieving points/nodes based on user query", expanded = False) as status:
            for movie_dict in movie_list:
                st.json(json.dumps(movie_dict, indent=4))
        
        with st.status("Simple Method: Generating response based on Similarity Search + LLM Call", expanded = True) as status:
            prompt_template = f"""
                Using the below context information respond to the user query.
                context: '{movie_list}'
                query: '{user_query}'
                Response structure should look like this:
                *Detailed Response*
                
                *Relevant Details in Table Format*
                """
            llm_response = llm.complete(prompt_template)
            response_parts = llm_response.text.split('```')
            st.markdown(response_parts[0])