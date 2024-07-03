from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from dotenv import load_dotenv

# load the data
loader = CSVLoader(file_path="interview_questions.csv")
documents = loader.load()

print(documents[0])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)