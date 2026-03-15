import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

prompt = "Geef een creatieve, korte beschrijving van de oceaan in twee zinnen."

temperatures = [0.0, 0.5, 1.0]

for temp in temperatures:
    print(f"\n{'='*50}")
    print(f"Temperature: {temp}")
    print('='*50)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        temperature=temp,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print(response.content[0].text)
