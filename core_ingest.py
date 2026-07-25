from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


PDF_FOLDER = "knowledge_base"

VECTOR_DB = "vectorstore"


def create_vectorstore():

    print("Loading PDFs...")

    loader = PyPDFDirectoryLoader(PDF_FOLDER)

    documents = loader.load()

    print(f"{len(documents)} pages loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"{len(chunks)} chunks created.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(VECTOR_DB)

    print("Vector database created successfully.")


if __name__ == "__main__":

    create_vectorstore()