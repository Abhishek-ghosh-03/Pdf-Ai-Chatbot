import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplate import css, bot_template, user_template
def get_text_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_chunk_pdf(raw_text):
    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    text_splitted = splitter.split_text(raw_text)
    return text_splitted

def get_vectorstore_pdf(raw_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings()
    vectorstore = FAISS.from_texts(texts = raw_chunks ,embedding = embeddings)
    return vectorstore


def get_conversation_pdf(vectorstore):
    #Dependency issue
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history' , return_messages= True
    )
    conversational = ConversationalRetrievalChain.from_llm(llm=llm ,retriever=vectorstore.as_retriever(),memory=memory)
    return conversational


def handle_question(ask_question):
    response = st.session_state.conversation({"question" : ask_question})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace('{{MSG}}', message.content),unsafe_allow_html=True) 
            # leftover using the html-template here to replace
        else:
            st.write(bot_template.replace('{{MSG}}', message.content),unsafe_allow_html=True) 



def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple Pdf Conversational Tool",page_icon=":Books:")
    st.header("Multiple Pdf Conversational Tool")
    ask_question = st.text_input("Ask Your Questions?")
    if ask_question:
        handle_question(ask_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload the pdf's to be summarised and click on the process button",
                         accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #text
                raw_text = get_text_pdf(pdf_docs)
                #chunks
                raw_chunks = get_chunk_pdf(raw_text)
                # st.write(raw_chunks)
                #embedding
                vectorstore = get_vectorstore_pdf(raw_chunks)
                #conversational
                st.session_state.conversation = get_conversation_pdf(vectorstore)

if __name__ == '__main__':
    main()