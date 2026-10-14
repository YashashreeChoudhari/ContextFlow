
from types import SimpleNamespace
from unittest.mock import patch

from src.agent import run_agent


def make_response(content=None, tool_calls=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


def test_agent_revises_answer_and_verifies_again():
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="rag_search",
            arguments='{"question": "What are Python uses?"}',
        ),
    )

    api_responses = [
        make_response(tool_calls=[tool_call]),
        make_response(
            content=(
                "Python is used for web development. "
                "It is also the fastest programming language."
            )
        ),
    ]

    evidence = (
        "Python is commonly used for web development, "
        "data analysis, machine learning, automation, and scripting."
    )

    retrieved_results = [
        {
            "content": evidence,
            "metadata": {"source": "data/sample.md"},
        }
    ]

    verification_results = [
        {
            "supported": False,
            "explanation": (
                "The claim about being the fastest language "
                "is not supported by the evidence."
            ),
            "unsupported_claims": [
                "Python is the fastest programming language."
            ],
        },
        {
            "supported": True,
            "explanation": (
                "The revised answer is supported by the evidence."
            ),
            "unsupported_claims": [],
        },
    ]

    revised_answer = (
        "Python is commonly used for web development, data analysis, "
        "machine learning, automation, and scripting."
    )

    with (
        patch(
            "src.agent.rag_search",
            return_value=retrieved_results,
        ) as mock_search,
        patch(
            "src.agent.client.chat.completions.create",
            side_effect=api_responses,
        ) as mock_api,
        patch(
            "src.agent.verify_answer",
            side_effect=verification_results,
        ) as mock_verify,
        patch(
            "src.agent.revise_answer",
            return_value=revised_answer,
        ) as mock_revise,
    ):
        result = run_agent(
            "What are Python's common uses?"
        )

    mock_search.assert_called_once()
    assert mock_api.call_count == 2
    assert mock_verify.call_count == 2
    mock_revise.assert_called_once()

    assert revised_answer in result

    assert "data/sample.md" in result

    assert "Verification warning:" not in result

    print("\nRevision-loop test passed!")
