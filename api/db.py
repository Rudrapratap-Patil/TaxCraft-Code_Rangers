import os
from getpass import getpass

HF_TOKEN = getpass("HF Token:")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

qdrant_key = "<your-key>"
qdrant_url = "<your-url>"

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

