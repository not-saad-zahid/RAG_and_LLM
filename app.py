from langchain_community.callbacks import get_openai_callback
import streamlit as st
import yaml
import time
from dotenv import load_dotenv
from llm import get_llm
from embeddings import get_embedding_model
from rag_pipeline import build_rag

load_dotenv()

TOKEN_PRICING = {
    # OpenAI models
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-5-mini-2025-08-07": {"input": 0.25, "output": 2.00},
    "gpt-5-nano-2025-08-07": {"input": 0.05, "output": 0.40},
    
    # Anthropic Claude models
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    
    "default": {"input": 1.0, "output": 2.0}
}

def calculate_cost(total_tokens, model_name, input_ratio=0.6):
    """
    Calculate cost based on total tokens and model pricing.
    
    Args:
        total_tokens: Total number of tokens used
        model_name: Name of the LLM model
        input_ratio: Estimated ratio of input tokens (default 0.6 = 60% input, 40% output)
    
    Returns:
        Estimated cost in USD
    """
    # Find matching pricing (partial match on model name)
    pricing = TOKEN_PRICING.get("default")
    for model_key in TOKEN_PRICING:
        if model_key.lower() in model_name.lower():
            pricing = TOKEN_PRICING[model_key]
            break
    
    # Estimate input/output split
    input_tokens = total_tokens * input_ratio
    output_tokens = total_tokens * (1 - input_ratio)
    
    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


# Load config
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_rag():
    config = load_config()
    llm = get_llm(config)
    embeddings = get_embedding_model(config)
    rag = build_rag(llm, embeddings, config)
    return rag, config


def format_chat_history(chat_history, max_turns=5):
    """
    Convert last N messages into text for LLM context
    """

    history = chat_history[-max_turns*2:]  # last N exchanges

    formatted = ""

    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"

    return formatted


def is_small_talk(query: str) -> bool:
    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "how are you", "what's up"
    ]
    q = query.lower().strip()
    return any(q == g or q.startswith(g) for g in greetings)

def get_final_answer_streaming(rag, config, user_query, chat_history):
    """
    Generator function that yields tokens as they come from the LLM
    Yields: (current_text, similar_chunks)
    Returns: (final_text, similar_chunks, total_tokens) after streaming completes
    """
    if is_small_talk(user_query):
        answer = "Hello! 👋 How can I help you?"
        yield answer, []
        return answer, [], 0

    chat_context = format_chat_history(chat_history)
    enhanced_query = f"""
    Conversation so far:
    {chat_context}

    Current question:
    {user_query}
    """

    similar_chunks = []
    total_tokens = 0
    final_text = ""
    
    with get_openai_callback() as cb:
        try:
            # Stream without tokens initially
            for chunk in rag.stream({"query": enhanced_query}):
                final_text = chunk.get("answer", "")
                similar_chunks = chunk.get("source_documents", [])
                yield final_text, similar_chunks 
            
            total_tokens = cb.total_tokens
            # print(f"[DEBUG] Streaming complete. Tokens used: {total_tokens}")
            
        except (AttributeError, NotImplementedError, Exception) as e:
            # Fallback to non-streaming
            print(f"[DEBUG] Streaming failed: {e}. Falling back to non-streaming mode.")
            response = rag.invoke({"query": enhanced_query})
            similar_chunks = response.get("source_documents", [])
            total_tokens = cb.total_tokens
            
            # Simulate streaming
            result_text = response.get('result', '')
            final_text = result_text
            words = result_text.split() 
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                if i % 3 == 0 or i == len(words) - 1:
                    yield current_text.strip(), similar_chunks
                    time.sleep(0.05)
    
    # After streaming completes, yield one more time with tokens
    yield final_text, similar_chunks, total_tokens
    return final_text, similar_chunks, total_tokens

st.markdown("""
<style>

/* Header */
.app-header {
    position: sticky;
    padding: 15px;
    margin-bottom: 5px;
    top: 0;
}

/* Input bar */
.input-area {
    position: sticky;
    bottom: 0;
    background-color: #0E1117;
    padding-top: 10px;
}

</style>
""", unsafe_allow_html=True)



st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="centered"
)


st.markdown("""
<div class="app-header">
    <h2>📄 Document RAG Assistant</h2>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Chat container */
.chat-box {
    max-height: 70vh;
    overflow-y: auto;
    padding: 15px;
}

/* User message (right side) */
.user-msg {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    margin: 8px 0;
}

.user-icon {
    margin-left: 6px;
    font-size: 20px;
}

.user-bubble {
    background-color: #1f2933;
    color: #fff;
    padding: 6px 10px;
    border-radius: 10px;
    max-width: 70%;
    word-wrap: break-word;
}

/* Bot message (left side) */
.bot-msg {
    display: flex;
    justify-content: flex-start;
    align-items: flex-end;
    margin: 8px 0;
}

.bot-icon {
    margin-right: 6px;
    font-size: 20px;
}

.bot-bubble {
    background-color: #1f2933;
    color: #fff;
    padding: 8px 12px;
    border-radius: 10px;
    max-width: 70%;
    word-wrap: break-word;
}

</style>
""", unsafe_allow_html=True)



# ---------------------------
# Load RAG Once
# ---------------------------
if "rag" not in st.session_state:
    st.session_state["rag"], st.session_state["config"] = get_rag()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = 0
if "similar_chunks" not in st.session_state:
    st.session_state["similar_chunks"] = []


# ---------------------------
# Chat Display (Scrollable)
# ---------------------------
st.markdown('<div class="chat-box">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:

    if msg["role"] == "user":

        st.markdown(
            f"""
            <div class="user-msg">
                <div class="user-bubble">{msg["content"]}</div>
                <div class="user-icon">👤</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-msg">
                <div class="bot-icon">🤖</div>
                <div class="bot-bubble">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 📚 Show sources if present
        if "sources" in msg and msg["sources"]:

            top_sources = msg["sources"][:3]  # Show top 3 sources

            with st.expander("📚 Reference Chunks Used"):

                for i, doc in enumerate(top_sources):

                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")

                    with st.expander(f"Source {i+1}: {source} (Page {page})"):

                        st.write(doc.page_content)


st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Chat Input (Bottom)
# ---------------------------
prompt = st.chat_input("Ask something about your documents...")


if prompt and not st.session_state.get("processing", False):

    # Prevent double processing
    st.session_state["processing"] = True

    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    st.session_state["chat_input"] = ""

    # Refresh UI so user sees message
    st.rerun()

# 2️⃣ Generate bot response

if st.session_state.get("processing", False):
    last_user_msg = st.session_state.chat_history[-1]["content"]
    response_placeholder = st.empty()
    
    full_response = ""
    similar_chunks = []
    tokens_used = 0
    
    # Stream the response
    stream_gen = get_final_answer_streaming(
        st.session_state["rag"],
        st.session_state["config"],
        last_user_msg,
        st.session_state["chat_history"]
    )
    
    # Process streaming results
    try:
        while True:
            try:
                # Get next chunk
                result = next(stream_gen)
                
                # Handle different return signatures
                if len(result) == 2:
                    streamed_text, chunks = result
                elif len(result) == 3:
                    streamed_text, chunks, tokens = result
                    tokens_used = tokens  # Update tokens as they come
                
                full_response = streamed_text
                similar_chunks = chunks
                
                # Update display
                with response_placeholder.container():
                    st.markdown(
                        f"""
                        <div class="bot-msg">
                            <div class="bot-icon">🤖</div>
                            <div class="bot-bubble">{full_response}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            except StopIteration as e:
                # When generator completes, it returns the final values
                final_result = e.value
                if final_result and len(final_result) == 3:
                    full_response, similar_chunks, tokens_used = final_result
                break
                
    except Exception as e:
        st.error(f"Error during streaming: {e}")
    

    st.session_state["chat_history"] = st.session_state["chat_history"] + [{"role": "assistant", "content": full_response, "sources": similar_chunks}]
    st.session_state["total_tokens"] += tokens_used  
    st.session_state["similar_chunks"] = similar_chunks

    st.session_state["processing"] = False

    st.rerun()


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    
    st.markdown("### LLM being used")
    st.write(st.session_state["config"]['llm']['model_name'])
    
    st.markdown("---")
    
    st.markdown("### 💲 Cost Estimation")
    
    model_name = st.session_state["config"]['llm']['model_name']
    total_tokens = st.session_state["total_tokens"]
    estimated_cost = calculate_cost(total_tokens, model_name)
    
    if total_tokens > 0:
        st.write(f"**${estimated_cost:.6f}** USD")
        st.caption(f"Based on {total_tokens:,} tokens")
    else:
        st.write("**$0.000000** USD")
        st.caption("No tokens used yet")

    st.markdown("---")
        
    st.markdown("### 📊 Token Usage")
    st.write(st.session_state['total_tokens'])

    st.markdown("---")

    if st.button("🗑️ Reset Chat"):
        st.session_state["chat_history"] = []
        st.rerun()