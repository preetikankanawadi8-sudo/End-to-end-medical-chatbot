# app.py
from flask import Flask, render_template, request
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import *
import os

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)
load_dotenv()

# Embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

# Connect to Pinecone Index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -------------------- FREE LOCAL LLM (FLAN-T5) --------------------

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{context}\n\nUser Question: {input}")
])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"input": RunnablePassthrough(),
     "context": retriever | format_docs}
    | prompt
    | llm
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke(msg)
    print("Response:", response)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)