from fastapi import FastAPI, Form
import openai
from pydantic import BaseModel
import hashlib
from cachetools import LRUCache
import os

# Initialize FastAPI app
app = FastAPI()

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create an LRU (Least Recently Used) Cache in memory (max size of 100 items)
cache = LRUCache(maxsize=100)

def generate_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use the correct model here
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    # Access the 'choices' attribute properly
    return response.choices[0].message.content.strip()

def normalize_prompt(prompt: str) -> str:
    """
    Normalize the prompt by converting it to lowercase and stripping extra spaces.
    """
    return " ".join(prompt.lower().split())


def hash_prompt(prompt: str) -> str:
    """
    Hash the normalized prompt to create a unique cache key.
    """
    return hashlib.sha256(prompt.encode()).hexdigest()


@app.post("/chat")
async def chat(prompt: str = Form(...)):
    """
    Chat endpoint to handle user prompts.
    """
    prompt = prompt

    # Normalize and hash the prompt to generate a unique cache key
    normalized_prompt = normalize_prompt(prompt)
    cached_response = cache.get(hash_prompt(normalized_prompt))
    print("Cache: -", cache)

    if cached_response:
        # If a cached response exists, return it
        return {"response": cached_response, "cached": True}

    # Otherwise, generate a new response using OpenAI GPT
    response = generate_response(prompt)

    # Cache the response for future similar prompts
    cache[hash_prompt(normalized_prompt)] = response

    return {"response": response, "cached": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
