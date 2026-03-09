import anthropic
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class Analyse(BaseModel):
    samenvatting: str
    toon: str
    trefwoorden: List[str]
    sentiment: str


tekst = input("Voer een stuk tekst in om te analyseren:\n> ")

response = client.messages.parse(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=(
        "Je bent een tekstanalyse-assistent. Analyseer de gegeven tekst en geef altijd "
        "een gestructureerd antwoord terug met precies vier velden:\n"
        "- samenvatting: een korte samenvatting van de tekst\n"
        "- toon: de toon van de tekst (bijv. formeel, informeel, zakelijk, emotioneel, neutraal)\n"
        "- trefwoorden: een lijst van de belangrijkste trefwoorden\n"
        "- sentiment: exact 'positief', 'negatief' of 'neutraal'"
    ),
    messages=[{"role": "user", "content": tekst}],
    output_format=Analyse,
)

analyse = response.parsed_output

print("\n--- Analyse ---")
print(f"Samenvatting : {analyse.samenvatting}")
print(f"Toon         : {analyse.toon}")
print(f"Trefwoorden  : {', '.join(analyse.trefwoorden)}")
print(f"Sentiment    : {analyse.sentiment}")
