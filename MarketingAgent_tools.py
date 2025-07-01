import os
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter

serper = GoogleSerperAPIWrapper()

def get_file_tools():
    toolkit = FileManagementToolkit(root_dir=os.getenv("FILE_TOOL_ROOT", "reports"))
    return toolkit.get_tools()

async def other_tools() -> list[Tool]:
    file_tools = get_file_tools()
    search_tool = Tool(
        name="search",
        func=serper.run,
        description="Run a Google Serper web search"
    )
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
    return file_tools + [search_tool, wiki_tool]

rag_chroma_db = None
rag_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def ingest_blogs_for_rag(directory_path: str = "blog_corpus") -> str:
    global rag_chroma_db
    docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                docs.extend(rag_text_splitter.create_documents([text]))
    if not docs:
        return "No blog files found for ingestion."
    if rag_chroma_db is None:
        rag_chroma_db = Chroma.from_documents(docs, rag_embeddings, persist_directory="chroma_db")
    else:
        rag_chroma_db.add_documents(docs)
    return f"Ingested {len(docs)} blog chunks from '{directory_path}'."

def rag_retrieve(query: str, k: int = 4) -> str:
    if rag_chroma_db is None:
        return "No blog data ingested yet."
    docs = rag_chroma_db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

def rag_tools():
    return [
        Tool(
            name="ingest_blogs_for_rag",
            func=ingest_blogs_for_rag,
            description="Ingest all .txt/.md blog files from the blog_corpus directory for RAG."
        ),
        Tool(
            name="rag_retrieve",
            func=rag_retrieve,
            description="Retrieve relevant context from ingested blogs for a query. Input: query string."
        ),
    ]