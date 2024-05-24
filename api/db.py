import os
from getpass import getpass

HF_TOKEN = getpass("HF Token:")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

qdrant_key = "OUhqChVDKOQcNf3470y6yfJsImOqWcWANbb8I5EqZ25ARgsaPQIa3w"
qdrant_url = "https://719303b5-6c1c-481a-9db0-cf1d832ab5ce.us-east4-0.gcp.cloud.qdrant.io:6333"

from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Qdrant

WEBSITE_URL = "https://medium.com/@jaintarun7/multimodal-using-gemini-and-llamaindex-f622a190cc32"

data = WebBaseLoader(WEBSITE_URL)

docs = data.load()

docs[0].page_content

text_split = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)

chunks = text_split.split_documents(docs)

len(chunks)

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = HF_TOKEN, model_name = "BAAI/bge-base-en-v1.5" #thenlper/gte-large
)

vector_db = Qdrant.from_documents(
    chunks,embeddings, path="./db",collection_name="gemini")
search = vector_db.similarity_search("what Gemini model is used in the blog article")
search
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs = {"k":2}
)
retriever.get_relevant_documents("what Gemini model is used in the blog article")
llm = HuggingFaceHub(
    repo_id = "huggingfaceh4/zephyr-7b-alpha",
    model_kwargs = {"temperature":0.1,"max_new_tokens":512,"return_full_text":False}
)
query = "what is llm framework used in the blog"
prompt = f"""
<|system|>
You are an AI assitant that follow instructions extremely well. Please be truthful and give direct answers
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)
response = qa.run(prompt)
response
