__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop["pysqlite3"]
# from dotenv import load_dotenv
# load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
#from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile, os

#제목
st.title("chatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Loader
    # loader = PyPDFLoader("example.pdf")
    # pages = loader.load_and_split()

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #embedding
    embeddings_model = OpenAIEmbeddings()

    #load it into chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에 질문해보세요:")
    question = st.text_input("질문을 입력하세요")
    
    if question:
        with st.spinner('PDF에 물어보는 중...'):
            llm = ChatOpenAI(temperature=0)

            qa_chain = RetrievalQA.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=db.as_retriever(),
            )
            result = qa_chain({"query": question})
            st.write(result["result"])
    
    # llm = ChatOpenAI(temperature=0)
    # # retriever_from_llm = MultiQueryRetriever.from_llm(
    # #     retriever=db.as_retriever(), llm=llm
    # # )
    # # docs = retriever_from_llm.get_relevant_documents(query=question)
    # # print(len(docs))
    # # print(docs)

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     chain_type="stuff",
    #     retriever=db.as_retriever(),
    # )
    # result = qa_chain({"query": question})
    # print(result)
