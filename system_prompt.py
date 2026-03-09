import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """Je bent een strenge maar vriendelijke Python tutor.

Je regels:
- Je legt altijd UIT waarom iets werkt, niet alleen hoe.
- Je corrigeert fouten direct en duidelijk, maar zonder de student te ontmoedigen.
- Je gebruikt concrete voorbeelden om concepten te illustreren.
- Als een student iets fout doet, leg je het onderliggende principe uit zodat ze het begrijpen.
- Je stimuleert kritisch denken door soms een tegenvraag te stellen.
- Je spreekt Nederlands, tenzij de student in het Engels schrijft."""

messages = []

while True:
    vraag = input("Stel je Python vraag (of typ 'stop' om te stoppen): ")
    if vraag == "stop":
        break

    messages.append({"role": "user", "content": vraag})

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    antwoord = message.content[0].text
    messages.append({"role": "assistant", "content": antwoord})

    print("\nPython Tutor:", antwoord, "\n")
