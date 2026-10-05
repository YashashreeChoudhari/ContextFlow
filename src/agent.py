import os
import time

from dotenv import load_dotenv
from google import genai

from src.tools import rag_search

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

def run_agent(question: str):
    results = rag_search(question)

    if not results:
        return "I could not find relevant information in the knowledge base."

    context = "\n\n".join(
        result["content"]
        for result in results
    )

    prompt = f"""
You are ContextFlow, an AI research agent.

Use the retrieved knowledge below to answer the user's question.

Retrieved knowledge:
{context}

User question:
{question}

Give a clear and concise answer.
Do not invent information.
"""

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )

            return response.text

        except Exception as error:
            print(
                f"\nGemini request failed "
                f"(attempt {attempt}/{max_attempts})"
            )
            print(f"Error: {error}")

            if attempt < max_attempts:
                print("Retrying in 20 seconds...")
                time.sleep(20)
            else:
                raise

if __name__ == "__main__":
    question = input("Ask ContextFlow: ").strip()

    if not question:
        raise SystemExit("Please enter a question.")

    print("\nSearching knowledge base...")

    answer = run_agent(question)

    print("\nAnswer:\n")
    print(answer)