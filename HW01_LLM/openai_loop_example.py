"""
Jednoducha ukazka LLM API s vlastnim toolem.

Do souboru .env ve stejne slozce uloz:
    OPENAI_API_KEY=tvuj_api_klic

Spusteni:
    uv --cache-dir .uv-cache run python openai_loop_example.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).with_name(".env"))

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

DEBUG = True


tools = [
    {
        "type": "function",
        "function": {
            "name": "square_number",
            "description": "Vrati druhou mocninu zadaneho cisla.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "Cislo, ktere se ma umocnit na druhou.",
                    },
                },
                "required": ["number"],
            },
        },
    }
]


def square_number(number):
    return number * number


for number in range(1, 4):
    print(f"\n================ Cislo {number} ================")

    messages = [
        {
            "role": "developer",
            "content": "Odpovidej cesky, strucne a slusne.",
        },
        {
            "role": "user",
            "content": (
                f"Rekni kratky vtip s cislem {number}. "
                "Nejdriv pouzij tool square_number a ve vtipu pouzij vysledek."
            ),
        },
    ]

    if DEBUG:
        print("\n[1] Posilam modelu prompt a dostupny tool:")
        print(messages[-1]["content"])
        print("Tool:", tools[0]["function"]["name"])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "square_number"},
        },
    )

    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    square = square_number(arguments["number"])
    tool_output = {
        "number": arguments["number"],
        "square": square,
    }

    if DEBUG:
        print("\n[2] Model si vyzadal tool call:")
        print("Tool call ID:", tool_call.id)
        print("Nazev toolu:", tool_call.function.name)
        print("Argumenty od modelu:", tool_call.function.arguments)
        print("\n[3] Python spustil lokalni funkci:")
        print(f"square_number({arguments['number']}) -> {square}")

    messages.append(
        {
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_output),
        }
    )

    if DEBUG:
        print("\n[4] Posilam vysledek toolu zpatky modelu:")
        print(messages[-1])

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    joke = final_response.choices[0].message.content

    print("\n[5] Finalni odpoved modelu:")
    print(f"--- Vtip cislo {number}: {number}^2 = {square} ---")
    print(joke)
