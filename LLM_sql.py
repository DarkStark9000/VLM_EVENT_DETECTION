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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from colorama import Fore
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Config --------------------
CHROMA_DB_PATH = "chroma_db"  # Persistent directory path for Chroma DB

# Open-source LLM Configuration
LLM_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"  # Efficient 3.8B parameter model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 4096

# Initialize open-source LLM
logger.info("Loading open-source LLM")
try:
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Create text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_length=MAX_LENGTH,
        temperature=0.7,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    logger.info("Open-source LLM loaded successfully")
except Exception as e:
    logger.error(f"Error loading LLM: {e}")
    logger.warning("Falling back to mock responses")
    text_generator = None

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
        logger.warning("No matching session_id found. Returning top-k results without filter")
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
    
# -------------------- RAG Prompt via googleapi --------------------

def ask_model(contexts, query, chat_history):
    """
    Generate response using open-source LLM with RAG context
    """
    if text_generator is None:
        # Fallback response if model not available
        return f"I understand you're asking about: {query}. However, the language model is currently unavailable. Based on the retrieved context, I can see information about video segments, but cannot provide a detailed analysis at the moment."
    
    context_text = "\n---\n".join([
        f"[{m['video_name']}] ({m['start_time']} - {m['end_time']}): {doc}"
        for m, doc in zip(contexts['metadatas'][0], contexts['documents'][0])
    ])

    # Build chat history context
    history_text = ""
    if chat_history:
        history_text = "\n".join([
            f"Previous Q: {msg.get('content', '')}" if msg.get('role') == 'user' else f"Previous A: {msg.get('content', '')}"
            for msg in chat_history[-4:]  # Last 4 messages for context
        ])

    # Build prompt for open-source LLM
    prompt = f"""<|system|>
You are an expert video content analyzer. Analyze video segments and answer questions based on the provided context. Be factual and concise.<|end|>

<|user|>
Context from video analysis:
{context_text}

Recent conversation history:
{history_text}

Question: {query}

Please provide a clear, factual answer based on the video content shown in the context. Include relevant timestamps when available.<|end|>

<|assistant|>"""

    try:
        # Generate response
        response = text_generator(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated_text = response[0]['generated_text']
        
        # Extract only the assistant's response
        if "<|assistant|>" in generated_text:
            answer = generated_text.split("<|assistant|>")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        logger.info(f"Generated response: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I encountered an error while processing your question about the video content. The question was: {query}"