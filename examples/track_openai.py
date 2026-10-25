"""Track the footprint of real OpenAI and Anthropic calls.

pip install green-ai openai anthropic
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...
python examples/track_openai.py
"""

from green_ai import Tracker

tracker = Tracker(zone="FRA")  # attribute to the French grid, or leave None for provider defaults

try:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarise the carbon footprint of LLM inference in two lines."}],
    )
    print(tracker.add(resp))
except Exception as exc:  # noqa: BLE001 - example script
    print("OpenAI call skipped:", exc)

try:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": "What is prefill in LLM inference?"}],
    )
    print(tracker.add(msg))
except Exception as exc:  # noqa: BLE001 - example script
    print("Anthropic call skipped:", exc)

print(tracker.summary())
