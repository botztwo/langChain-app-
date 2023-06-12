import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


with st.sidebar:
    st.title("Bennet's LLMChat APP")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made by Bennet Botz with refernece to (https://youtube.com/@engineerprompt)')


def main():
    st.header("Chat with PDF")

    load_dotenv()
    #code allowing a user to input a PDF 
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text = text)
    
        store_name = pdf.name[:-4]


        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            
        else:
            #embedd the tect to numerical vectors
            embeddings  = OpenAIEmbeddings()
            #organize the embeddings
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore,f)
            
        #Accepting User Input
        query = st.text_input("Ask Question about PDF")

        if query:
            docs = VectorStore.similarity_search(query=query, k = 3)

            #sending the top 3 similar PDF documents as well as the users query to the davinci model from OpenAi
            llm =OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                #generate a resposne
                response = chain.run(input_documents = docs, question = query )
                print(cb)
            #dispaly resposne 
            st.write(response)


    return None

if __name__ == '__main__':
    main()