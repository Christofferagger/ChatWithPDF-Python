from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_texts(chunks, embeddings)

    db1 = FAISS.from_texts(["foo"], embeddings)
    print(db1.docstore._dict)
    db2 = FAISS.from_texts(["baa"], embeddings)
    print(db2.docstore._dict)
    db1.merge_from(db2)
    print(db1.docstore._dict)

    return db

def main():
    
    st.title = ("ChatWithPDF")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    file = st.file_uploader("Upload your pdf", type="pdf", accept_multiple_files=True)

    if file is not None:
        text = ''
        pdf_read = PdfReader(file)
        for page in pdf_read.pages:
            text += page.extract_text()
        
        db = process_text(text)

        query = st.text_input("Ask a question?")

        context = "".join([f"{interaction['you']} {interaction['response']}" for interaction in st.session_state.chat_history])

        if query:

            prompt = db.similarity_search(query)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                if st.session_state.chat_history:
                    question = f"context: {context}\nQuestion: {query}"
                    response = chain.run(input_documents=prompt, question=question)
                    print(cost)
                else:
                    response = chain.run(input_documents=prompt, question=query)
                    print(cost)
            
            st.session_state.chat_history.append({'you': query, 'response': response})
            st.write(response)
            for interaction in st.session_state.chat_history:
                st.write(f"You: {interaction['you']}")
                st.write(f"Response: {interaction['response']}")

if __name__ == "__main__":
    main()


