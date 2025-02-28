import pathlib
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI
from dotenv import load_dotenv

# Initialize the OpenAI client - reuse the key that was loaded in main.py
# dotenv_path = pathlib.Path(__file__).parent.parent / '.env'
# load_dotenv(dotenv_path=dotenv_path)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate a chat completion using OpenAI's API.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        
        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        # Log the error here
        raise Exception(f"Error generating chat completion: {str(e)}")


async def analyze_text_with_query(query: str, text: str) -> Dict[str, Any]:
    """
    Analyze text based on a query using OpenAI's API.
    """
    prompt = f"""
    You are analyzing text data and need to answer the user's query.
    
    USER QUERY: {query}
    
    TEXT TO ANALYZE: 
    {text}
    
    Based on the text above, please answer the query. If the information isn't present
    in the text, please indicate that clearly.
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes document data."},
        {"role": "user", "content": prompt}
    ]
    
    return await generate_chat_completion(
        messages=messages,
        temperature=0.3,  # Lower temperature for more focused responses
    )