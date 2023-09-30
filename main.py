import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


load_dotenv()

system_template = """Use the following pieces of context to answer the users question. 
Whenever the question refers to you, you should consider that it refers to Tadej.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def main():
    st.title('Chat With Tadej 💬')
    st.subheader("I'm a 🤖 and have parsed Tadej's page https://krevh.com\nAsk me anything!")

    prompt = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Question", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        openai_embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            embedding_function=openai_embeddings,
            persist_directory=DB_DIR
        )
        collection = vectordb.get()

        if len(collection.get('documents', [])) == 0:
            loader = WebBaseLoader(['https://krevh.com', 'https://krevh.com/experience.html'])
            data = loader.load()

            text_splitter = CharacterTextSplitter(separator='\n',
                                                  chunk_size=500,
                                                  chunk_overlap=40)
            docs = text_splitter.split_documents(data)

            vectordb = Chroma.from_documents(
                docs,
                openai_embeddings,
                persist_directory=DB_DIR
            )
            vectordb.persist()

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        response = qa(prompt)
        st.write(response.get('result', 'Ups! Something went wrong...'))


if __name__ == '__main__':
    main()