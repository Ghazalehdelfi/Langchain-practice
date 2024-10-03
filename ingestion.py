import os
from dotenv import load_dotenv
# from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)

if __name__ == "__main__":
    print(os.environ["INDEX_NAME"])
    print("loading document...")
    loader = PyPDFLoader("dyson-logos-challenge-of-the-frog-idol.pdf")
    document = loader.load()
    
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting to vector db...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
