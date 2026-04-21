"""
Ukazka jednoduche smycky volani OpenAI LLM API s vlastnim toolem.

API klic uloz do souboru .env ve stejne slozce jako tento skript:
    OPENAI_API_KEY=tvuj_api_klic

Spusteni pres uv:
    uv --cache-dir .uv-cache run python openai_loop_example.py

"""

import json
from pathlib import Path

from openai import OpenAI


def load_api_key_from_env_file() -> str:
    env_path = Path(__file__).with_name(".env")

    if not env_path.exists():
        raise FileNotFoundError(f"Soubor s API klicem neexistuje: {env_path}")

    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY" and separator:
            api_key = value.strip().strip('"').strip("'")
            if api_key:
                return api_key

    raise ValueError("V souboru .env chybi vyplnena hodnota OPENAI_API_KEY.")


client = OpenAI(api_key=load_api_key_from_env_file())


TOOLS = [
    {
        "type": "function",
        "name": "square_number",
        "description": "Vrati druhou mocninu zadaneho celeho cisla.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "integer",
                    "description": "Cislo, ktere se ma umocnit na druhou.",
                },
            },
            "required": ["number"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def square_number(number: int) -> dict[str, int]:
    return {
        "number": number,
        "square": number**2,
    }


def get_function_calls(response) -> list:
    return [
        item
        for item in response.output or []
        if getattr(item, "type", None) == "function_call"
    ]


def handle_tool_call(function_call) -> dict:
    if function_call.name != "square_number":
        raise ValueError(f"Neznamy tool: {function_call.name}")

    arguments = json.loads(function_call.arguments or "{}")
    result = square_number(int(arguments["number"]))

    return {
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": json.dumps(result, ensure_ascii=False),
    }


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
        prompt = (
            f"Rekni kratky vtip s cislem {number}. "
            "Pred napsanim vtipu zavolej tool square_number a ve vtipu pouzij "
            "vysledek druhe mocniny."
        )

        tool_response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            instructions="Odpovidej cesky, strucne a slusne.",
            input=prompt,
            tools=TOOLS,
            tool_choice={"type": "function", "name": "square_number"},
            parallel_tool_calls=False,
            max_output_tokens=500,
        )

        function_calls = get_function_calls(tool_response)
        if not function_calls:
            print(f"\n--- Vtip cislo {number} ---")
            print("Model nevyzadal zadny tool call.")
            print(f"Status odpovedi: {tool_response.status}")
            continue

        tool_outputs = [handle_tool_call(function_calls[0])]
        tool_result = json.loads(tool_outputs[0]["output"])

        final_response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            instructions="Odpovidej cesky, strucne a slusne.",
            input=tool_outputs,
            previous_response_id=tool_response.id,
            tools=TOOLS,
            tool_choice="none",
            max_output_tokens=500,
        )

        joke = extract_text(final_response)

        print(f"\n--- Vtip cislo {number}: {number}^2 = {tool_result['square']} ---")
        if joke:
            print(joke)
        else:
            print("Model nevratil zadny viditelny text.")
            print(f"Status odpovedi: {final_response.status}")
            if final_response.incomplete_details:
                print(f"Detail: {final_response.incomplete_details}")


if __name__ == "__main__":
    main()
