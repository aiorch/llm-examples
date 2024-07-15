from langchain.llms import HuggingFaceTextGenInference
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredMarkdownLoader

# LangChain LLM Ingetration with TGI
llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8080/",
    max_new_tokens=4096,
    top_k=50,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

# Embeddings
# model_name = "BAAI/bge-base-en-v1.5"
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Store
url = "http://localhost:6333"

client = QdrantClient("localhost", port=6333)
vectorstore = Qdrant(
    client=client,
    collection_name="docs",
    embeddings=embeddings,
)


# RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_type="mmr", k=20)
)

# Prompt
# Optionally, pull from the Hub
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")
# Or, define your own:

# Zephyr 7B Prompt Template
template = """
<|system|>
You are a Terraform code expert.
Answer the question based only on the following context:
{context}
</s>
<|user|>
Question: {question}
</s>
<|assistant|>
"""

# Code Llama Prompt Template
# template = """
# [INST] <<SYS>>
# You are a Terraform expert.
# <</SYS>>
# Answer the question based only on the following context:
# {context}
# Question: {question}
# [/INST]
# """

prompt = ChatPromptTemplate.from_template(template)

model = llm

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str

chain = chain.with_types(input_type=Question)
