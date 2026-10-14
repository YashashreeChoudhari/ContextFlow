
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.verification import verify_answer


def test_verification_accepts_supported_answer():
    evidence = (
        "Python is commonly used for web development "
        "and data analysis."
    )
    answer = (
        "Python is commonly used for web development "
        "and data analysis."
    )

    mock_result = {
        "supported": True,
        "explanation": "Both claims are supported by the evidence.",
        "unsupported_claims": [],
    }

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(mock_result)
                )
            )
        ]
    )

    with patch(
        "src.verification.client.chat.completions.create",
        return_value=mock_response,
    ):
        result = verify_answer(answer, evidence)

    assert result["supported"] is True
    assert result["unsupported_claims"] == []
    assert "supported" in result["explanation"].lower()


def test_verification_rejects_unsupported_answer():
    evidence = "Python is commonly used for web development."
    answer = "Python is the fastest programming language."

    mock_result = {
        "supported": False,
        "explanation": "The evidence does not establish this claim.",
        "unsupported_claims": [
            "Python is the fastest programming language."
        ],
    }

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(mock_result)
                )
            )
        ]
    )

    with patch(
        "src.verification.client.chat.completions.create",
        return_value=mock_response,
    ):
        result = verify_answer(answer, evidence)

    assert result["supported"] is False
    assert len(result["unsupported_claims"]) == 1
