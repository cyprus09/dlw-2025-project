"""Service for interacting with OpenAI's API for text analysis and structured data extraction."""

import logging
import os
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

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
    """Generate a chat completion using OpenAI's API."""
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
                "total_tokens": response.usage.total_tokens,
            },
        }
    except Exception as e:
        # Log the error here
        import traceback

        error_traceback = traceback.format_exc()
        print(f"Error in generate_chat_completion: {str(e)}\n{error_traceback}")
        raise Exception(f"Error generating chat completion: {str(e)}")


async def analyze_text_with_query(query: str, text: str) -> Dict[str, Any]:
    """Analyze text based on a query using OpenAI's API."""
    prompt = f"""
    You are analyzing text data and need to answer the user's query.

    USER QUERY: {query}

    TEXT TO ANALYZE:
    {text}

    Based on the text above, please answer the query. If the information isn't present
    in the text, please indicate that clearly.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes document data.",
        },
        {"role": "user", "content": prompt},
    ]

    return await generate_chat_completion(
        messages=messages,
        temperature=0.3,  # Lower temperature for more focused responses
    )


async def analyze_text_with_schema(
    text: str, schema: Type[BaseModel]
) -> Dict[str, Any]:
    """Analyze text based on a provided schema type using OpenAI's beta.chat.completions.parse method.

    Args:
        text: The text to analyze
        schema: The schema model class to use for structured data extraction

    Returns:
        Dictionary containing the structured response and usage statistics
    """
    logging.info(f"Starting analyze_text_with_schema with schema: {schema.__name__}")
    logging.info(f"Text length: {len(text)} characters")

    try:
        logging.info("Preparing to call OpenAI API with parse method")
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that extracts structured data from carbon offset project documents. "
                        "Please follow the given ProjectAnalysisSchema exactly. Output valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze the following text (a project description or PDF OCR). "
                        f"Return data in JSON that fits the ProjectAnalysisSchema:\n\n{text}"
                    ),
                },
            ],
            response_format=schema,
            temperature=0.3,  # Lower temperature for more focused responses
        )

        logging.info("Successfully received response from OpenAI API")
        logging.info(
            f"Token usage - Prompt: {completion.usage.prompt_tokens}, Completion: {completion.usage.completion_tokens}, Total: {completion.usage.total_tokens}"
        )

        # Get the parsed response
        parsed_data = completion.choices[0].message.parsed
        logging.info("Successfully parsed response data")

        logging.info("Returning structured response")
        return {
            "structured_response": parsed_data.model_dump(),
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
        }
    except Exception as e:
        # Log the error here
        import traceback

        error_traceback = traceback.format_exc()
        logging.error(f"Error in analyze_text_with_schema: {str(e)}")
        logging.error(f"Traceback: {error_traceback}")
        print(f"Error in analyze_text_with_schema: {str(e)}\n{error_traceback}")
        raise Exception(f"Error generating structured output: {str(e)}")
