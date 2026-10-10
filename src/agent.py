
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.tools import rag_search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODEL = "openrouter/free"

tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the ContextFlow knowledge base for evidence relevant to a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to search for."
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }
    }
]


def run_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are ContextFlow, a research assistant. "
                "Use rag_search when the question requires the knowledge base. "
                "Base answers on retrieved evidence and do not invent facts. "
                "If the evidence is insufficient, say so."
            )
        },
        {"role": "user", "content": question}
    ]

    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="required",
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content or "No answer was returned."

        for tool_call in message.tool_calls:
            if tool_call.function.name != "rag_search":
                continue

            arguments = json.loads(tool_call.function.arguments)
            results = rag_search(arguments["question"])

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(results, ensure_ascii=False)
            })

    return "I couldn't complete the request within the tool-call limit."


if __name__ == "__main__":
    question = input("Ask ContextFlow: ").strip()

    if not question:
        raise SystemExit("Please enter a question.")

    try:
        print("\nRunning ContextFlow agent...\n")
        print(run_agent(question))
    except Exception as error:
        print(f"Agent error: {error}")
