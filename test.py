#업로드 문제해결
__import__('pysqlite3')
import sys
sys.modules['pysqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
from streamlit_extras.buy_me_a_coffee import button
from langchain_community.chat_models import ChatAnthropic


#수익화 함수
def BMC():
    button(username="silverstone", floating=True, width=221)

BMC()

# stream 함수
class StreamChain:
    def __init__(self, chain):
        self.chain = chain

    def stream(self, query):
        response = self.chain.stream(query)
        complete_response = ""
        for token in response:
            print(token, end="", flush=True)
            complete_response += token
        return complete_response

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("Choose a file", type=['.pdf'])
st.write("---")

#업로드 함수
def pdf_to_document(uploaded_file) :
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    pass
    
#Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()

# load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!!")
    question = st.text_input('질문을 입력하세요')
    if st.button('질문하기') :
        with st.spinner('질문하는중...'):
            chat_box = st.empty()
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            # 스트림
            #chat = ChatAnthropic(model="gpt-3.5-turbo",)
            #for chunk in chat.stream(qa_chain):
            #    print(chunk.content, end="", flush=True)
            result = qa_chain({"query": question})
            st.write(result["result"])
