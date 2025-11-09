from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

pinecone_api_key = os.getenv("pinecone_api_key")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")


extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(minimal_docs)
embeddings = download_embeddings()



pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot-test"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)



docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,   
    index_name=index_name,
)

