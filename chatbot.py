import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

def load_documents(path="data/faqs.txt"):
    loader = TextLoader(path)
    return loader.load()

def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        model="llama3-8b-8192",
        temperature=0.3
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return chain

def main():
    print("🤖 FAQ Chatbot (Groq) ready! Type your question or 'exit' to quit.")

    documents = load_documents()
    vectorstore = create_vector_store(documents)
    qa_chain = create_qa_chain(vectorstore)

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = qa_chain.invoke({"query": query})
        print(f"Bot: {response['result']}\n")

if __name__ == "__main__":
    main()
