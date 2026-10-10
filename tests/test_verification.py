from src.verification import verify_answer

evidence = """
Python is commonly used for web development, data analysis,
machine learning, automation, and scripting.
"""

answer = (
    "Python is commonly used for web development "
    "and data analysis."
)

result = verify_answer(answer, evidence)

print("\nVerification result:")
print(result)
