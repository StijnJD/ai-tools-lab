import anthropic
from dotenv import load_dotenv
import os

load_dotenv("keys.env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Leg uit wat een API is in 2 zinnen."}
    ]
)

print(message.content[0].text)

