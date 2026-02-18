from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import pipeline

def get_llm(config):
    provider = config["llm"]["provider"]

    if provider == "huggingface":
        pipe = pipeline(
            "text-generation",
            model=config["llm"]["model_name"],
            max_new_tokens=config["llm"]["max_tokens"],
            temperature=config["llm"]["temperature"]
        )
        return HuggingFacePipeline(pipeline=pipe)

    elif provider == "openai":
        return ChatOpenAI(
            model=config["llm"]["model_name"],
            streaming=True,
            temperature=config["llm"]["temperature"]
        )

    else:
        raise ValueError("Unsupported LLM provider")
