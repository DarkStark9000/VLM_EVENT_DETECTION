# from chromadb import PersistentClient
# from sentence_transformers import SentenceTransformer
# import requests
# import json
# from colorama import Fore
# from follow_code.chat_history_db import load_chat_history  # Make sure this file is present

# # -------------------- Config --------------------
# RUNPOD_API_URL = "https://eapq37g3s8oqcm-8080.proxy.runpod.net/v1/chat/completions"
# MODEL_NAME = "gemma-3-27b-it"
# CHROMA_DB_PATH = "chroma_db"  # Persistent directory path for Chroma DB

# # -------------------- Init Embedding & Vector DB --------------------
# embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
# chroma_client = PersistentClient(path=CHROMA_DB_PATH)

# # -------------------- Query & Retrieve --------------------
# # def retrieve_context(query, collection_name, k=3):
# #     collection = chroma_client.get_or_create_collection(name=collection_name)
# #     embedding = embedder.encode([query])[0]
# #     results = collection.query(query_embeddings=[embedding], n_results=k)
# #     return results

# def retrieve_context(query, collection_name, session_id, k=3):
#     collection = chroma_client.get_or_create_collection(name=collection_name)
#     embedding = embedder.encode([query])[0]

#     results = collection.query(
#         query_embeddings=[embedding],
#         n_results=k,
#         where={"session_id": session_id}  # Filter by session_id in metadata
#     )
#     return results


# # -------------------- RAG Prompt via RunPod --------------------
# def ask_model(contexts, query, chat_history=None):
#     # Format context as readable text with timestamps
#     context_text = "\n---\n".join([
#         f"[{m['video_name']}] ({m['start_time']} - {m['end_time']}): {doc}"
#         for m, doc in zip(contexts['metadatas'][0], contexts['documents'][0])
#     ])

#     # Extract video paths for reference
#     video_path = [m['video_path'] for m in contexts['metadatas'][0]]

#     # Add previous chat rounds if available
#     chat_messages = chat_history if chat_history else []
#     chat_messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"})

#     # Create payload for RunPod API
#     payload = {
#         "model": MODEL_NAME,
#         "messages": chat_messages,
#         "max_tokens": 8096,
#         "temperature": 0.2,
#         "top_p": 0.9
#     }

#     headers = {
#         "Content-Type": "application/json"
#     }

#     # Send POST request
#     response = requests.post(RUNPOD_API_URL, headers=headers, json=payload)

#     if response.status_code == 200:
#         res_json = response.json()
#         return res_json['choices'][0]['message']['content'], video_path
#     else:
#         raise Exception(f"RunPod API error: {response.status_code} - {response.text}")


from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import json
from colorama import Fore
from openai import OpenAI

# -------------------- Config --------------------
# RUNPOD_API_URL = "https://l84r74zg5i8ndw-8080.proxy.runpod.net/v1/chat/completions"
# MODEL_NAME = "gemma-3-27b-it"
CHROMA_DB_PATH = "chroma_db"  # Persistent directory path for Chroma DB


#for gemma api google api 


# === CONFIG ===
# vLLM OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"  # vLLM doesn't require real API key
)
MODEL_NAME = "Meta-Llama/Meta-Llama-3.1-8B-Instruct"

# -------------------- Init Embedding & Vector DB --------------------
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
    
    # If nothing matches, fallback to unfiltered top-k
    if not filtered_results:
        print(" No matching session_id found. Returning top-k results without filter.")
        filtered_results = list(zip(results['documents'][0][:k], results['metadatas'][0][:k]))

    return {
        "documents": [[doc for doc, _ in filtered_results]],
        "metadatas": [[meta for _, meta in filtered_results]]
    }


# -------------------- Query & Retrieve --------------------
# def retrieve_context(query, collection_name, k=3):
#     collection = chroma_client.get_or_create_collection(name=collection_name)
#     embedding = embedder.encode([query])[0]
#     results = collection.query(query_embeddings=[embedding], n_results=k)
#     return results

# -------------------- RAG Prompt via RunPod --------------------
# def ask_model(contexts, query):
#     # Format context as readable text with timestamps
#     context_text = "\n---\n".join([
#         f"[{m['video_name']}] ({m['start_time']} - {m['end_time']}): {doc}"
#         for m, doc in zip(contexts['metadatas'][0], contexts['documents'][0])
#     ])

#     # Extract video paths for reference
#     video_path = [m['video_path'] for m in contexts['metadatas'][0]]

#     # Build prompt
#     prompt = f"""
#     You are an expert video content analyzer.

#     Analyze the following context segments retrieved from videos. Your task is to answer the question strictly based on the content, without generating random or speculative answers.

#     - Provide a concise summary in paragraph form.
#     - If multiple segments refer to the same video name, combine their information into a single response.
#     - Mention the relevant timestamps, if available.
#     - Rephrase the final answer according to the language used in the original query.

#     Ensure that your response is factual, clearly structured, and reflects only what is observed or stated in the video content.

#     Context:
#     {context_text}

#     Question: {query}
#     Answer:
# """

#     # Create payload for RunPod API
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "max_tokens": 8096,
#         "temperature": 0.2,
#         "top_p": 0.9
#     }

#     headers = {
#         "Content-Type": "application/json"
#     }

#     # Send POST request
#     response = requests.post(RUNPOD_API_URL, headers=headers, json=payload)

#     if response.status_code == 200:
#         res_json = response.json()
#         return res_json['choices'][0]['message']['content'], video_path
#     else:
#         raise Exception(f"RunPod API error: {response.status_code} - {response.text}")
    
# -------------------- RAG Prompt via vLLM --------------------

def ask_model(contexts, query, chat_history):
    context_text = "\n---\n".join([
        f"[{m['video_name']}] ({m['start_time']} - {m['end_time']}): {doc}"
        for m, doc in zip(contexts['metadatas'][0], contexts['documents'][0])
    ])

    # # Extract video paths for reference
    # video_path = [m['video_path'] for m in contexts['metadatas'][0]]

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
    
    print(response.choices[0].message.content)
    return response.choices[0].message.content
