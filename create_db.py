import os
import logging

import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

log = logging.getLogger(__name__)

logging.basicConfig(
    format="%(levelname)s : %(message)s", level=logging.INFO, force=True
)

NAMESPACE = "model-series"
QUERY_TEXT = "what is the inspection time interval ?"
FILE = "/media/llt/LENOVO5/data/IR/c2b0c2b0c2b0-cessna_182s_1997on_mm_182smm.pdf"

# initialize pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV"]
)

# split documents
# by default, text splitter works page per page (separator "\n\n"),
# you can use another separator as argument
# loader = PyPDFLoader(FILE)
# text_splitter = CharacterTextSplitter(chunk_size=1000,
#                                       chunk_overlap=100)
# loader = PyPDFLoader(FILE)
# documents = loader.load_and_split(text_splitter=text_splitter)

# Select embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

index_name = "demo-ir"

# db = Pinecone.from_documents(documents,
#                              embeddings,
#                              index_name=index_name,
#                              namespace=NAMESPACE)

# if you already have an index, you can load it like this
db = Pinecone.from_existing_index(index_name,
                                  embeddings,
                                  namespace=NAMESPACE)

# index_description = pinecone.describe_index(index_name)
# index = pinecone.Index(index_name)
# index_stats_response = index.describe_index_stats()

# Create retriever interface
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"]),
                                 chain_type='stuff',
                                 retriever=retriever)
result_ = qa.run(QUERY_TEXT)

best_scores = db.similarity_search_with_score(QUERY_TEXT,
                                              k=2,
                                              namespace=NAMESPACE)

# delete
# index = pinecone.Index(index_name)
# delete_response = index.delete(namespace=NAMESPACE, delete_all=True)