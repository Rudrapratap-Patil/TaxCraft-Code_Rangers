import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pdfplumber
import openai
import pandas as pd

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title('Ask me')

# Add iframe code with CSS styling for alignment
st.write("""
<div style="float: right; width: 70%;">
    <iframe src="https://www.chatbase.co/chatbot-iframe/1fKcG2dmj7pMK0VHkV1xF" title="Chatbot" width="100%" style="height: 100%; min-height: 700px" frameborder="0"></iframe>
</div>
""", unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_tables_with_pdfplumber(uploaded_file):
    tables = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table[1:], columns=table[0])  # Create DataFrame, assuming first row as header
                tables.append(df)
    return tables

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def answer_question(user_question, pdf_text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(pdf_text)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        all_text = ''
        for page in pdf.pages:
            all_text += page.extract_text()
    return all_text

def summarize_text_with_gpt3(text):
    response = openai.Completion.create(
        model="text-davinci-004",
        prompt=f"Summarize the following text in a point-wise format:\n{text}",
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def generate_tax_reduction_suggestions_with_ai(summarized_text):
    prompt = f"""
    Given the following summarized details of the tax document:
    {summarized_text}

    Provide detailed and personalized comments on how to reduce the tax liability based on the provided document context and general tax-saving strategies.
    """
    response = openai.Completion.create(
        model="text-davinci-004",
        prompt=prompt,
        max_tokens=250,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def textReader():
    st.title('TaxCraft')
    st.header("Chat with TaxCraft")
    st.write("Upload the documents related to tax and get information on tax deduction")

    uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

    user_question = st.text_input("Ask a Question:")

    if uploaded_files and user_question:
        pdf_text = get_pdf_text(uploaded_files)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                answer = answer_question(user_question, pdf_text)
                st.success("Done")
                st.write("Reply: ", answer)

        st.header('Summary and Analysis')
        extracted_text = extract_text_from_pdf(uploaded_files[0])
        summarized_text = summarize_text_with_gpt3(extracted_text)
        st.write('Summary of your document:', summarized_text)
        
        tables = extract_tables_with_pdfplumber(uploaded_files[0])
        st.header('Extracted Tables')
        for i, table in enumerate(tables):
            st.write(f'Table {i+1}')
            st.dataframe(table)

        suggestions = generate_tax_reduction_suggestions_with_ai(summarized_text)
        st.header('Tax Reduction Suggestions')
        if suggestions:
            st.write(suggestions)
        else:
            st.write('No specific suggestions based on the provided data.')

    st.sidebar.title('Contents')
    st.sidebar.write('[Home](#)')
    st.sidebar.write('[Tax Calculator](#tax)')
    st.sidebar.write('[Investment Suggestions](#suggestions)')
    st.sidebar.write('[About Us](#about)')
    st.markdown('---')



    
st.write('Code Rangers Team© 2024')

# Run the app
if __name__ == '__main__':
    textReader()
