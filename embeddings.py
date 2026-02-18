from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import  OpenAIEmbeddings

def get_embedding_model(config):
    provider = config["embeddings"]["provider"]

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=config["embeddings"]["model_name"]
        )

    elif provider == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    else:
        raise ValueError("Unsupported embedding provider")
