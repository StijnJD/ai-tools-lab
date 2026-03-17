import argparse
import sys
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "Je bent een professionele samenvattingsassistent. Analyseer de aangeleverde tekst en geef "
    "een gestructureerde, professionele samenvatting terug met de volgende onderdelen:\n\n"
    "1. **Samenvatting** – Een beknopte samenvatting van 2-4 zinnen.\n"
    "2. **Hoofdpunten** – De 3-7 belangrijkste punten als genummerde lijst.\n"
    "3. **Conclusies** – De belangrijkste conclusies of takeaways.\n\n"
    "Schrijf helder, zakelijk en to the point. Gebruik dezelfde taal als de brontekst."
)


def extraheer_tekst_uit_pdf(pad: str) -> str:
    try:
        import pypdf
    except ImportError:
        print(
            "Fout: pypdf is niet geïnstalleerd. Voer uit: pip install pypdf",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        reader = pypdf.PdfReader(pad)
    except pypdf.errors.PdfReadError as e:
        print(f"Fout: kan PDF niet lezen — {e}", file=sys.stderr)
        sys.exit(1)

    paginas = []
    for pagina in reader.pages:
        tekst = pagina.extract_text()
        if tekst:
            paginas.append(tekst)

    if not paginas:
        print("Fout: geen tekst gevonden in de PDF.", file=sys.stderr)
        sys.exit(1)

    return "\n\n".join(paginas)


def vat_samen(tekst: str, bestandsnaam: str) -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"Samenvatting van: {bestandsnaam}\n")
    print("=" * 60)

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Vat de volgende tekst samen:\n\n{tekst}",
            }
        ],
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maak een professionele samenvatting van een PDF-bestand."
    )
    parser.add_argument("pdf", help="Pad naar het PDF-bestand")
    args = parser.parse_args()

    pad = args.pdf

    if not os.path.isfile(pad):
        print(f"Fout: bestand niet gevonden: {pad}", file=sys.stderr)
        sys.exit(1)

    if not pad.lower().endswith(".pdf"):
        print("Waarschuwing: bestand heeft geen .pdf-extensie.", file=sys.stderr)

    tekst = extraheer_tekst_uit_pdf(pad)
    bestandsnaam = os.path.basename(pad)

    try:
        vat_samen(tekst, bestandsnaam)
    except anthropic.AuthenticationError:
        print(
            "Fout: ongeldige of ontbrekende API-sleutel. Controleer je ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)
    except anthropic.RateLimitError:
        print(
            "Fout: API-limiet bereikt. Wacht even en probeer het opnieuw.",
            file=sys.stderr,
        )
        sys.exit(1)
    except anthropic.BadRequestError as e:
        print(f"Fout: ongeldig verzoek — {e.message}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError:
        print(
            "Fout: geen verbinding met de API. Controleer je internetverbinding.",
            file=sys.stderr,
        )
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(
            f"Fout: API-fout (HTTP {e.status_code}) — {e.message}", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
