
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

CHROMA_DB_PATH = "chroma_db"
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"  # vLLM doesn't require real API key
)
MODEL_NAME = "Meta-Llama/Meta-Llama-3.1-8B-Instruct"


embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
chroma_client = PersistentClient(path=CHROMA_DB_PATH)

def retrieve_context(query, collection_name, session_id, k=3):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    embedding = embedder.encode([query])[0]
    
    # Query more results to allow filtering
    results = collection.query(query_embeddings=[embedding], n_results=20)  
    
    # Filter by session_id from metadata
    filtered_results = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        if meta.get("session_id") == session_id:
            filtered_results.append((doc, meta))
            if len(filtered_results) >= k:
                break
    
    if not filtered_results:
        filtered_results = list(zip(results['documents'][0][:k], results['metadatas'][0][:k]))

    return {
        "documents": [[doc for doc, _ in filtered_results]],
        "metadatas": [[meta for _, meta in filtered_results]]
    }



def ask_model(contexts, query, chat_history):
    context_text = "\n---\n".join([
        f"[{m['video_name']}] ({m['start_time']} - {m['end_time']}): {doc}"
        for m, doc in zip(contexts['metadatas'][0], contexts['documents'][0])
    ])

    # Build prompt
    prompt = f"""
    You are an expert video content analyzer.

    Analyze the following context segments retrieved from videos. Your task is to answer the question strictly based on the content, without generating random or speculative answers.

    - Provide a concise summary in paragraph form.
    - If multiple segments refer to the same video name, combine their information into a single response.
    - Mention the relevant timestamps, if available.
    - Rephrase the final answer according to the language used in the original query.

    Ensure that your response is factual, clearly structured, and reflects only what is observed or stated in the video content.
    


    Context:
    {context_text}

    chat_histroy : {chat_history}

    Question: {query}
    Answer:
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
        temperature=0.2,
        top_p=0.9
    )
    

    return response.choices[0].message.content
