
import os
import json
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.tools import rag_search
from src.verification import verify_answer


# ==================================================
# 1. ENVIRONMENT SETUP
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from your .env file."
    )


# ==================================================
# 2. OPENROUTER CLIENT
# ==================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=60.0,
    max_retries=2,
)

MODEL = "openrouter/free"
MAX_REVISION_ATTEMPTS = 2


# ==================================================
# 3. DEFINE THE RAG TOOL
# ==================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the ContextFlow knowledge base for relevant "
                "evidence and source metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to search for.",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    }
]


# ==================================================
# 4. GENERATE OR REVISE AN ANSWER
# ==================================================

def revise_answer(
    question: str,
    answer: str,
    evidence: str,
    explanation: str,
    unsupported_claims: list,
) -> str:
    """Revise an answer using verifier feedback and existing evidence."""

    feedback = json.dumps(
        {
            "explanation": explanation,
            "unsupported_claims": unsupported_claims,
        },
        ensure_ascii=False,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the critic-and-revision component of "
                    "ContextFlow. Improve the draft answer using only "
                    "the supplied evidence. Remove or qualify claims "
                    "that the evidence does not support. Do not add "
                    "outside facts, invented citations, or new claims. "
                    "If evidence is insufficient, say so clearly. "
                    "Return only the revised answer, not your analysis "
                    "or a JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"AVAILABLE EVIDENCE:\n{evidence}\n\n"
                    f"CURRENT ANSWER:\n{answer}\n\n"
                    f"VERIFIER FEEDBACK:\n{feedback}\n\n"
                    "Revise the answer to address the verifier feedback."
                ),
            },
        ],
        temperature=0,
    )

    if not response.choices:
        raise RuntimeError(
            "OpenRouter returned no choices during revision."
        )

    revised = (response.choices[0].message.content or "").strip()

    if not revised:
        raise RuntimeError(
            "The revision model returned an empty answer."
        )

    return revised


# ==================================================
# 5. RUN THE AGENT AND RETRIEVE EVIDENCE
# ==================================================

def run_agent(question: str) -> str:
    """Run RAG, generate an answer, verify it, and revise if needed."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are ContextFlow, a research assistant. "
                "Use rag_search to retrieve relevant knowledge-base "
                "evidence before answering. Base factual claims on "
                "retrieved evidence. Do not invent facts or citations. "
                "If the evidence is insufficient, say so clearly. "
                "Give a clear, concise answer."
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

    # --------------------------------------------------
    # Agent tool-calling loop
    # --------------------------------------------------

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

        if not tool_calls:
            answer = (message.content or "").strip()

            if not answer:
                raise RuntimeError(
                    "The model returned an empty answer. "
                    "Try again or choose another model."
                )

            break

        messages.append(message)

        for tool_call in tool_calls:
            if tool_call.function.name != "rag_search":
                tool_output = json.dumps({
                    "error": "Unknown tool requested."
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

                    print(f"[Tool] Searching: {search_question}")

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

    # --------------------------------------------------
    # Prepare evidence for verification
    # --------------------------------------------------

    evidence = "\n\n".join(
        dict.fromkeys(evidence_parts)
    ).strip()

    # --------------------------------------------------
    # 6. CRITIC-AND-REVISION LOOP
    # --------------------------------------------------

    if not evidence:
        print(
            "\n[Verification skipped] "
            "No evidence was retrieved."
        )

        answer += (
            "\n\nVerification warning: No retrieved evidence was "
            "available to verify this answer."
        )

    else:
        verification_failed = False
        verified = False

        # Initial verification plus at most two revisions.
        for attempt in range(MAX_REVISION_ATTEMPTS + 1):
            stage = (
                "Initial check"
                if attempt == 0
                else f"Check after revision {attempt}"
            )

            print(f"\n[Verification] {stage}...")

            try:
                result = verify_answer(
                    answer=answer,
                    evidence=evidence,
                )

                if not isinstance(result, dict):
                    raise ValueError(
                        "Verifier did not return a dictionary."
                    )

                supported = result.get("supported")
                explanation = result.get("explanation")
                unsupported_claims = result.get(
                    "unsupported_claims"
                )

                if not isinstance(supported, bool):
                    raise ValueError(
                        "Verifier's supported field must be boolean."
                    )

                if not isinstance(explanation, str):
                    raise ValueError(
                        "Verifier's explanation must be a string."
                    )

                if not isinstance(unsupported_claims, list):
                    raise ValueError(
                        "Verifier's unsupported_claims must be a list."
                    )

                print("[Verification explanation]", explanation)

            except Exception as error:
                print(
                    f"\n[Verification error] "
                    f"{type(error).__name__}: {error}"
                )
                traceback.print_exc()

                verification_failed = True
                break

            if supported:
                print(
                    "[Verification status] "
                    "Supported by retrieved evidence."
                )
                verified = True
                break

            print(
                "[Verification status] "
                "Unsupported claims detected."
            )
            print("[Unsupported claims]", unsupported_claims)

            # Do not revise again after the retry limit.
            if attempt >= MAX_REVISION_ATTEMPTS:
                print("[Revision] Maximum attempts reached.")
                break

            print(
                f"\n[Critic] Revising answer "
                f"(attempt {attempt + 1}/"
                f"{MAX_REVISION_ATTEMPTS})..."
            )

            try:
                answer = revise_answer(
                    question=question,
                    answer=answer,
                    evidence=evidence,
                    explanation=explanation,
                    unsupported_claims=unsupported_claims,
                )

                print("[Critic] Revision complete.")

            except Exception as error:
                print(
                    f"\n[Revision error] "
                    f"{type(error).__name__}: {error}"
                )
                traceback.print_exc()
                verification_failed = True
                break

        if verification_failed:
            answer += (
                "\n\nVerification warning: The verification or "
                "revision process failed. Do not treat this answer "
                "as verified."
            )

        elif not verified:
            answer += (
                "\n\nVerification warning: The answer still contains "
                "claims that could not be supported after the "
                "allowed revision attempts."
            )

    # --------------------------------------------------
    # 7. APPEND SOURCE FILENAMES
    # --------------------------------------------------

    sources = list(dict.fromkeys(retrieved_sources))

    if sources:
        answer += "\n\nSources:\n"
        answer += "\n".join(
            f"- {source}" for source in sources
        )

    return answer


# ==================================================
# 8. APPLICATION ENTRY POINT
# ==================================================

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
