import streamlit as st 
st.set_page_config("Chat PDF")

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a helpful tutor AI who only uses the context below to answer questions. 
    Be friendly, clear, and concise. If the answer is not in the context, respond with: 
    "I'm sorry, I couldn't find that in the provided document."

    IMPORTANT FORMATTING GUIDELINES:
    - When presenting education information, use clear formatting with bullet points or numbered lists
    - Separate different educational qualifications with clear visual breaks
    - Use consistent formatting for dates, institutions, and grades/scores
    - Make it easy to distinguish between different education levels
    - For resumes/CVs, maintain professional formatting while being readable

    Context:
    {context}

    Question:
    {question}

    Answer:
    """


    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db=FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write(response["output_text"])


def main():
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

                uploaded_items = [pdf.name for pdf in pdf_docs]
                st.session_state.processed_files.extend(uploaded_items)
                st.session_state.processed_files = list(set(st.session_state.processed_files))  # removes duplicates

    if st.session_state.processed_files:
        st.markdown("### ✅ Processed Files:")
        for file in st.session_state.processed_files:
            st.markdown(f"- {file}")

if __name__ == "__main__":
    main()