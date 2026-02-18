from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retriever import get_retriever

def build_rag(llm, embeddings, config):
    input_path = config["embeddings"]["input_path"]
    persist_directory = config["vector_store"]["persist_directory"]

    # Try to load existing Chroma DB, else create and persist
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if len(vectordb.get()["ids"]) == 0:
        documents = PyPDFDirectoryLoader(input_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=30)
        docs = splitter.split_documents(documents)
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

    retriever = get_retriever(vectordb, config)
    
    SYSTEM_PROMPT = """
        You are a professional AI assistant.

        Answer using ONLY the given information.

        Rules:
        - If the answer is missing, say: "The answer is not available in the provided documents."
        - Do not hallucinate.
        - Be concise and clear.
        - Do not mention sources or context.
        - Keep tone neutral and professional.
        - Use bullet points when helpful.
    """

    # Prompt template for the LLM
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template= SYSTEM_PROMPT + """
        Context: {context}
        Question: {question}
        Answer:
        """
    )
    
    # Format documents function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build a streaming-compatible RAG chain using LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    class StreamingRAGChain:
        def __init__(self, chain, retriever):
            self.chain = chain
            self.retriever = retriever
            
        def invoke(self, inputs):
            """For non-streaming calls"""
            query = inputs.get("query", inputs)
            
            # Get source documents
            source_docs = self.retriever.invoke(query)
            
            # Get answer
            result = self.chain.invoke(query)
            
            return {
                "result": result,
                "source_documents": source_docs
            }
        
        def stream(self, inputs):
            """For streaming calls"""
            query = inputs.get("query", inputs)
            
            # Get source documents first
            source_docs = self.retriever.invoke(query)
            
            # Stream the answer
            full_answer = ""
            for chunk in self.chain.stream(query):
                full_answer += chunk
                yield {
                    "answer": full_answer,
                    "source_documents": source_docs
                }
    
    return StreamingRAGChain(rag_chain, retriever)
