import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hallo!"}]
)
print(response.usage.input_tokens)
print(response.usage.output_tokens)