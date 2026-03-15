import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

prompt = "Leg uit hoe het internet werkt, in gewone taal."

print("Claude: ", end="", flush=True)

with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=512,
    messages=[
        {"role": "user", "content": prompt}
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()
