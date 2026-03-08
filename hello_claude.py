import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

while True:
    vraag = input("Stel je vraag (of typ 'stop' om te stoppen): ")
    if vraag == "stop":
        break
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": vraag}
        ]
    )
    print("\nClaude:", message.content[0].text, "\n")



