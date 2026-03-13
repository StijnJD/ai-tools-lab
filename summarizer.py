import sys
import anthropic
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List, Literal

load_dotenv()


class Samenvatting(BaseModel):
    bullet_points: List[str]  # exact 3 items
    toon: str
    trefwoorden: List[str]   # exact 5 items
    sentiment: Literal["positief", "negatief", "neutraal"]


def analyseer_tekst(tekst: str) -> Samenvatting:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "Je bent een tekstanalyse-assistent. Analyseer de gegeven tekst en geef een "
            "gestructureerd antwoord terug met precies vier velden:\n"
            "- bullet_points: een lijst van PRECIES 3 bullet points die de tekst samenvatten\n"
            "- toon: de toon van de tekst (bijv. formeel, informeel, zakelijk, emotioneel, neutraal)\n"
            "- trefwoorden: een lijst van PRECIES 5 trefwoorden\n"
            "- sentiment: exact 'positief', 'negatief' of 'neutraal'"
        ),
        messages=[{"role": "user", "content": tekst}],
        output_format=Samenvatting,
    )

    return response.parsed_output


def druk_resultaat_af(analyse: Samenvatting) -> None:
    print("\n--- Samenvatting ---")
    for i, punt in enumerate(analyse.bullet_points, 1):
        print(f"  {i}. {punt}")
    print(f"\nToon        : {analyse.toon}")
    print(f"Trefwoorden : {', '.join(analyse.trefwoorden)}")
    print(f"Sentiment   : {analyse.sentiment}")


def main() -> None:
    print("Plak hieronder de tekst die je wilt analyseren.")
    print("Sluit af met een lege regel + Enter (of Ctrl+D):\n")

    regels = []
    try:
        while True:
            regel = input()
            if regel == "" and regels:
                break
            regels.append(regel)
    except EOFError:
        pass

    tekst = "\n".join(regels).strip()

    if not tekst:
        print("Fout: geen tekst ingevoerd.", file=sys.stderr)
        sys.exit(1)

    try:
        analyse = analyseer_tekst(tekst)
        druk_resultaat_af(analyse)

    except anthropic.AuthenticationError:
        print("Fout: ongeldige of ontbrekende API-sleutel. Controleer je ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)
    except anthropic.RateLimitError:
        print("Fout: API-limiet bereikt. Wacht even en probeer het opnieuw.", file=sys.stderr)
        sys.exit(1)
    except anthropic.BadRequestError as e:
        print(f"Fout: ongeldig verzoek — {e.message}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("Fout: geen verbinding met de API. Controleer je internetverbinding.", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(f"Fout: API-fout (HTTP {e.status_code}) — {e.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
