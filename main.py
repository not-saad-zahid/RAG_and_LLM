import yaml
from llm import get_llm
from embeddings import get_embedding_model
from rag_pipeline import build_rag
from dotenv import load_dotenv
load_dotenv()

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Show which LLM and embedding model are used
llm_provider = config['llm']['provider']
llm_model = config['llm']['model_name']
emb_provider = config['embeddings']['provider']
emb_model = config['embeddings']['model_name']
print(f"LLM Provider: {llm_provider}, Model: {llm_model}")
print(f"Embedding Provider: {emb_provider}, Model: {emb_model}")

llm = get_llm(config)
embeddings = get_embedding_model(config)

# Build RAG
rag = build_rag(llm, embeddings, config)


# Track and print cumulative token count for OpenAI and HuggingFace LLMs
llm_type = llm_provider.lower()
total_tokens = 0
tokenizer = None
if llm_type == "huggingface":
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
    except Exception:
        tokenizer = None


def ask_and_count(query, total_tokens):
    """Non-streaming version (backward compatible)"""
    if llm_type == "openai":
        from langchain_community.callbacks import get_openai_callback
        similar_chunks = []

        try:
            with get_openai_callback() as cb:

                # Call chain (adjust key if needed)
                response = rag.invoke({"query": query})

                similar_chunks = response.get("source_documents", [])

                total_tokens += cb.total_tokens

                print(f"Tokens used: {cb.total_tokens}")
                print(f"Similar Chunks Retrieved: {len(similar_chunks)}")

        except ImportError:
            # Callback not available (non-OpenAI models)
            response = rag.invoke({"query": query})

            if isinstance(response, dict):
                similar_chunks = response.get("source_documents", [])

            print("Token tracking not supported for this LLM.")

        return response, total_tokens, similar_chunks


    elif llm_type == "huggingface":
        response = rag.invoke(query)
        if tokenizer:
            input_tokens = len(tokenizer.encode(query))
            # Try to extract output text from response
            output_text = ""
            if isinstance(response, list) and len(response) > 0:
                if hasattr(response[0], 'page_content'):
                    output_text = response[0].page_content
                elif isinstance(response[0], dict) and 'page_content' in response[0]:
                    output_text = response[0]['page_content']
                else:
                    output_text = str(response[0])
            else:
                output_text = str(response)
            output_tokens = len(tokenizer.encode(output_text))
            total_tokens += input_tokens + output_tokens
        return response, total_tokens, []
    else:
        response = rag.invoke(query)
        return response, total_tokens, []


def ask_and_stream(query, total_tokens):
    """Streaming version (NEW!)"""
    if llm_type == "openai":
        from langchain_community.callbacks import get_openai_callback
        similar_chunks = []
        
        print("\n🔄 Streaming response:")
        print("-" * 50)
        
        try:
            with get_openai_callback() as cb:
                full_response = ""
                
                # Stream the response
                for chunk in rag.stream({"query": query}):
                    answer = chunk.get("answer", "")
                    similar_chunks = chunk.get("source_documents", [])
                    
                    # Print only the new part
                    new_text = answer[len(full_response):]
                    print(new_text, end="", flush=True)
                    full_response = answer
                
                print("\n" + "-" * 50)
                total_tokens += cb.total_tokens
                
                print(f"\n✅ Streaming complete!")
                print(f"Tokens used: {cb.total_tokens}")
                print(f"Similar Chunks Retrieved: {len(similar_chunks)}")
                
                return {"result": full_response, "source_documents": similar_chunks}, total_tokens, similar_chunks
                
        except Exception as e:
            print(f"\n❌ Streaming failed: {e}")
            print("Falling back to invoke()...")
            return ask_and_count(query, total_tokens)
    else:
        # For non-OpenAI models, use invoke
        print("⚠️  Streaming not available for this model, using invoke()")
        return ask_and_count(query, total_tokens)


# Example usage: ask multiple queries
queries = [
    "What are the internships and projects done by the person?",
]

for q in queries:
    print(f"\n📝 Query: {q}")
    print("-" * 50)
    response, total_tokens, similar_chunks = ask_and_count(q, total_tokens)
    print(f"Response:\n{response['result']}")
    
    print("\n" + "=" * 60)
    
# Show similar chunks if needed
# print("\n📚 Similar Chunks:")
# for idx, chunk in enumerate(similar_chunks):
#     print(f"\nChunk {idx + 1}:")
#     print(chunk.page_content)
#     print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
#     print(f"Page: {chunk.metadata.get('page', 'N/A')}")


print(f"\n\n📊 Total tokens used: {total_tokens}")
