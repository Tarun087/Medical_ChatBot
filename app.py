from flask import Flask,render_template,jsonify,request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import *

import os

load_dotenv()

app = Flask(__name__)

pinecone_api_key = os.getenv("pinecone_api_key")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")



embeddings = download_embeddings()
index_name = "medical-chatbot-test"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatGroq(model=groq_model, api_key=groq_api_key, temperature=0)




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

def rag(question:str):
    results = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in results])
    chain = prompt | llm
    return chain.invoke({"context": context, "input": question})




@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(f"User question: {input}")
    response = rag(input)
    print(f"Chatbot answer: {response.content}")

    return str(response.content)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)