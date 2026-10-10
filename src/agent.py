
import os
import json
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.tools import rag_search
from src.verification import verify_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from your .env file."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=60.0,
    max_retries=2,
)

MODEL = "openrouter/free"


tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the ContextFlow knowledge base and return "
                "relevant text with source metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The question to search for in the "
                            "knowledge base."
                        ),
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    }
]

def run_agent(question: str) -> str:
    """Retrieve evidence, generate an answer, and verify it."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are ContextFlow, a research assistant. "
                "Use rag_search to find relevant information before "
                "answering. Base factual claims on retrieved evidence. "
                "Do not invent facts or source filenames. "
                "If evidence is insufficient, say so clearly. "
                "Answer in clear, concise language."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    evidence_parts = []
    retrieved_sources = []
    answer = None

    for iteration in range(5):
        print(f"\n[Agent] Round {iteration + 1}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        if not response.choices:
            raise RuntimeError(
                "OpenRouter returned no response choices."
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # The model has produced its final answer.
        if not tool_calls:
            answer = (message.content or "").strip()

            if not answer:
                raise RuntimeError(
                    "The model returned an empty answer. "
                    "Try again or select another model."
                )

            break

        messages.append(message)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            if tool_name != "rag_search":
                tool_output = json.dumps({
                    "error": f"Unknown tool: {tool_name}"
                })

            else:
                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                    search_question = arguments.get("question")

                    if (
                        not isinstance(search_question, str)
                        or not search_question.strip()
                    ):
                        raise ValueError(
                            "rag_search requires a non-empty question."
                        )

                    print(
                        f"[Tool] Searching: {search_question}"
                    )

                    results = rag_search(search_question)

                    print(
                        f"[Tool] Retrieved {len(results)} result(s)."
                    )

                    for result in results:
                        if not isinstance(result, dict):
                            continue

                        content = result.get("content", "")
                        metadata = result.get("metadata") or {}

                        if isinstance(content, str) and content.strip():
                            evidence_parts.append(content)

                        source = metadata.get("source")

                        if source:
                            retrieved_sources.append(str(source))

                    tool_output = json.dumps(
                        results,
                        ensure_ascii=False,
                        default=str,
                    )

                except Exception as error:
                    print(
                        f"[Tool error] {type(error).__name__}: {error}"
                    )

                    tool_output = json.dumps({
                        "error": (
                            "Knowledge-base search failed. "
                            "Do not invent search results."
                        )
                    })

            # Return the tool result to the model.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output,
            })

    else:
        raise RuntimeError(
            "The agent reached its tool-call limit without "
            "producing a final answer."
        )

    unique_evidence = list(dict.fromkeys(evidence_parts))
    evidence = "\n\n".join(unique_evidence).strip()

    if evidence:
        print("\n[Verification] Checking answer against evidence...")

        try:
            verification = verify_answer(
                answer=answer,
                evidence=evidence,
            )

            # Validate the verifier's output before trusting it.
            if not isinstance(verification, dict):
                raise ValueError(
                    "Verifier did not return a dictionary."
                )

            supported = verification.get("supported")
            explanation = verification.get("explanation")
            unsupported_claims = verification.get(
                "unsupported_claims"
            )

            if not isinstance(supported, bool):
                raise ValueError(
                    "Verifier's 'supported' value must be a boolean."
                )

            if not isinstance(explanation, str):
                raise ValueError(
                    "Verifier's 'explanation' must be a string."
                )

            if not isinstance(unsupported_claims, list):
                raise ValueError(
                    "Verifier's 'unsupported_claims' must be a list."
                )

            print(
                "\n[Verification explanation]",
                explanation,
            )

            if supported:
                print(
                    "[Verification status] "
                    "Supported by retrieved evidence."
                )

            else:
                print(
                    "[Verification status] "
                    "Unsupported claims detected."
                )

                print(
                    "[Unsupported claims]",
                    unsupported_claims,
                )

                answer += (
                    "\n\nVerification warning: Some claims could not "
                    "be supported by the retrieved evidence. "
                    "Please review the answer."
                )

        except Exception as error:
            print(
                "\n[Verification error]",
                f"{type(error).__name__}: {error}",
            )

            traceback.print_exc()

            answer += (
                "\n\nVerification warning: The answer could not be "
                "verified. Do not treat verification as successful."
            )

    else:
        print(
            "\n[Verification skipped] "
            "No evidence was retrieved."
        )

        answer += (
            "\n\nVerification warning: No retrieved evidence was "
            "available to verify this answer."
        )

    sources = list(dict.fromkeys(retrieved_sources))

    if sources:
        answer += "\n\nSources:\n"
        answer += "\n".join(
            f"- {source}" for source in sources
        )

    return answer

def main():
    print("=" * 48)
    print("          ContextFlow Research Agent")
    print("=" * 48)

    try:
        question = input("\nAsk ContextFlow: ").strip()

        if not question:
            print("Please enter a question.")
            return

        print("\nRunning ContextFlow...\n")

        answer = run_agent(question)

        print("\n" + "=" * 48)
        print("ANSWER")
        print("=" * 48)
        print(answer)

    except KeyboardInterrupt:
        print("\nAgent stopped.")

    except Exception as error:
        print(
            f"\n[Agent error] {type(error).__name__}: {error}"
        )
        traceback.print_exc()


if __name__ == "__main__":
    main()
