import os

import tempfile
import s3fs
import pinecone
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pypdfium2 as pdfium

NAMESPACE = "model-series"
INDEX_NAME = "demo-ir"

# Page title
st.set_page_config(page_title="Ask A Doc")
st.title("Ask A Doc")

st.write("MODEL 182/T182 SERIES 1997 AND ON")

st.cache_data()
def download_file():
    # download pdf file from s3
    remote_file_addr = st.secrets["REMOTE_FILE_ADDR"]
    temp_dir = os.path.join(tempfile.gettempdir(), "ir-demo")
    os.makedirs(temp_dir, exist_ok=True)
    local_file_addr = os.path.join(temp_dir, os.path.basename(remote_file_addr))
    fs = s3fs.S3FileSystem(anon=True)
    fs.download(remote_file_addr, local_file_addr)
    return local_file_addr

local_file_addr = download_file()

query_text = st.text_input('Enter your question:')

result = []
with st.form('myform', clear_on_submit=True):
    # init vectordb and retriever. why does this have to be here ?
    # if not, there is a pinecone connection error...
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENV"]
    )
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    db = Pinecone.from_existing_index(INDEX_NAME,
                                        embeddings,
                                        namespace=NAMESPACE)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"]),
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True)
    # submit question
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            response = qa(query_text)
            result.append(response)

if len(result):
    st.info(response["result"])
    st.write("The answer is based on following sources : ")
    # get source
    page_indices_ = [int(elem.metadata["page"]) for elem in response["source_documents"]]
    # remove duplicates
    page_indices = []
    for elem in page_indices_:
        if elem not in page_indices:
            page_indices.append(elem)
    # display pages
    pdf = pdfium.PdfDocument(local_file_addr)
    renderer = pdf.render(pdfium.PdfBitmap.to_pil, page_indices=page_indices)
    for image in renderer:
        st.image(image)
