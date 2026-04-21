"""
Ukazka jednoduche smycky volani OpenAI LLM API.

Poznamka: ChatGPT predplatne a OpenAI API jsou oddelene veci.
Pro spusteni skriptu potrebujes API klic v promenne prostredi OPENAI_API_KEY.
Klic si muzes ulozit do souboru .env ve stejne slozce jako tento skript.

Instalace:
    pip install openai

Spusteni:
    python openai_loop_example.py
"""

import os
from pathlib import Path

from openai import OpenAI


def load_local_env() -> None:
    env_path = Path(__file__).with_name(".env")

    if "OPENAI_API_KEY" in os.environ or not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY" and separator:
            os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
            return


load_local_env()
client = OpenAI()


def extract_text(response) -> str:
    if response.output_text:
        return response.output_text.strip()

    text_parts = []
    for item in response.output or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                text_parts.append(text)

    return "\n".join(text_parts).strip()


def main() -> None:
    for number in range(1, 6):
        prompt = f"Rekni kratky vtip s cislem {number}."

        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            instructions="Odpovidej cesky, strucne a slusne.",
            input=prompt,
            max_output_tokens=500,
        )

        joke = extract_text(response)

        print(f"\n--- Vtip cislo {number} ---")
        if joke:
            print(joke)
        else:
            print("Model nevratil zadny viditelny text.")
            print(f"Status odpovedi: {response.status}")
            if response.incomplete_details:
                print(f"Detail: {response.incomplete_details}")


if __name__ == "__main__":
    main()
