import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# De tekst die we elke keer analyseren
TEKST = """
Gisteren heb ik een presentatie gegeven voor mijn team over het nieuwe project.
Ik ben begonnen met wat achtergrond, daarna heb ik de planning uitgelegd.
Sommige mensen leken afgeleid. De vragen aan het einde waren oké.
Ik denk dat het redelijk ging, maar ik weet het niet zeker.
"""

SEPARATOR = "-" * 60


def vraag_claude(prompt: str) -> str:
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        result = stream.get_final_message()
    return result.content[0].text


# --- Versie 1: vage, minimale prompt ---
prompt_v1 = f"Geef feedback op deze tekst:\n\n{TEKST}"

# --- Versie 2: duidelijkere taak met context ---
prompt_v2 = f"""Je bent een communicatiecoach. Analyseer de onderstaande tekst en geef feedback.
Focus op: wat gaat goed, wat kan beter, en geef 1-2 concrete tips.

Tekst:
{TEKST}"""

# --- Versie 3: gestructureerde prompt met rol, formaat en voorbeeldoutput ---

prompt_v3 = f"""Je bent een ervaren communicatiecoach die professionals helpt hun presentatievaardigheden te verbeteren.

Analyseer de onderstaande zelfreflectie van iemand na een teampresentatie. Geef feedback in dit formaat:

**Sterke punten:** (wat deed de persoon goed?)
**Verbeterpunten:** (welke specifieke aspecten kunnen beter?)
**Blinde vlekken:** (wat ziet de persoon zelf over het hoofd?)
**Concrete actiepunten:** (2-3 specifieke stappen voor de volgende presentatie)

Wees eerlijk maar constructief. Baseer je alleen op wat er in de tekst staat.

Zelfreflectie:
{TEKST}"""


# --- Versie 4: few-shot prompting met een voorbeeld van input en output ---
VOORBEELD_TEKST = """
Ik heb gisteren een vergadering geleid met drie collega's over de budgetplanning.
Ik had een agenda gemaakt en die hebben we keurig afgewerkt.
Iedereen was op tijd en niemand had vragen, dus ik denk dat alles duidelijk was.
"""

VOORBEELD_OUTPUT = """**Sterke punten:** Je hebt de vergadering gestructureerd voorbereid (agenda) en strak geleid — dat zorgt voor efficiëntie en geeft deelnemers houvast.

**Verbeterpunten:** "Niemand had vragen" is niet per se positief; het kan betekenen dat deelnemers niet betrokken waren of de ruimte niet voelden om iets te zeggen.

**Blinde vlekken:** Je meet succes aan het volgen van de agenda, maar niet aan de uitkomst. Zijn er besluiten genomen? Weten deelnemers wat ze moeten doen? Dat ontbreekt in de reflectie.

**Concrete actiepunten:**
1. Stel aan het einde van elke vergadering expliciet de vraag: "Wat neem jij hieruit mee?" — dit activeert deelnemers en toetst begrip.
2. Noteer na afloop 2-3 concrete besluiten of acties, zodat je bij de volgende sessie kunt checken of ze zijn uitgevoerd.
3. Vraag één deelnemer om anonieme feedback — één zin over wat beter kon."""

prompt_v4 = f"""Je bent een ervaren communicatiecoach. Je taak: analyseer een zelfreflectie na een werkpresentatie of vergadering en geef feedback in een vast format.

Hieronder zie je een voorbeeld van hoe je dat doet.

---
VOORBEELD INPUT:
{VOORBEELD_TEKST.strip()}

VOORBEELD OUTPUT:
{VOORBEELD_OUTPUT.strip()}
---

Gebruik exact hetzelfde format voor de volgende zelfreflectie. Wees specifiek en baseer je alleen op wat er staat.

INPUT:
{TEKST.strip()}

OUTPUT:"""


# --- Versie 6: Chain of thought — stap voor stap redeneren voor de conclusie ---
prompt_v6 = f"""Je bent een ervaren communicatiecoach. Analyseer de onderstaande zelfreflectie na een teampresentatie.

Werk dit stap voor stap uit voordat je je eindoordeel geeft:

**Stap 1 — Wat staat er letterlijk in de tekst?**
Vat samen wat de persoon beschrijft, zonder interpretatie.

**Stap 2 — Wat zegt dit over de presentatie?**
Wat kunnen we afleiden uit de gekozen woorden, toon en wat wél/niet wordt vermeld?

**Stap 3 — Wat ontbreekt of wordt vermeden?**
Welke vragen beantwoordt de reflectie niet? Wat lijkt de persoon niet te zien?

**Stap 4 — Conclusie en advies**
Geef op basis van je redenering hierboven: 1 sterk punt, 1 blinde vlek, en 2 concrete actiepunten.

Zelfreflectie:
{TEKST}"""


# --- Versie 5: XML-tagged prompt voor duidelijke structuur ---
prompt_v5 = f"""<task>
  Analyseer de onderstaande zelfreflectie van iemand na een teampresentatie en geef gestructureerde feedback.
</task>

<role>
  Je bent een ervaren communicatiecoach die professionals helpt hun presentatievaardigheden te verbeteren.
  Wees eerlijk maar constructief. Baseer je alleen op wat er in de tekst staat.
</role>

<input>
  <zelfreflectie>
    {TEKST.strip()}
  </zelfreflectie>
</input>

<output_format>
  Geef je feedback in de volgende secties:
  - **Sterke punten:** wat deed de persoon goed?
  - **Verbeterpunten:** welke specifieke aspecten kunnen beter?
  - **Blinde vlekken:** wat ziet de persoon zelf over het hoofd?
  - **Concrete actiepunten:** 2-3 specifieke stappen voor de volgende presentatie
</output_format>"""


if __name__ == "__main__":
    versies = [
        ("Versie 1 — Vage prompt", prompt_v1),
        ("Versie 2 — Duidelijkere prompt met context", prompt_v2),
        ("Versie 3 — Gestructureerde prompt met rol en formaat", prompt_v3),
        ("Versie 4 — Few-shot prompting met voorbeeld", prompt_v4),
        ("Versie 5 — XML-tagged prompt", prompt_v5),
        ("Versie 6 — Chain of thought", prompt_v6),
    ]

    for titel, prompt in versies:
        print(f"\n{'=' * 60}")
        print(f"  {titel}")
        print(f"{'=' * 60}")
        print(f"\n[PROMPT]\n{prompt.strip()}\n")
        print(SEPARATOR)
        print("[OUTPUT]")
        print(vraag_claude(prompt))
        print()
