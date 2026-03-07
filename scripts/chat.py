#!/usr/bin/env python3
"""
Interactive terminal chat with Rahul's RAG Voicebot.
Usage: python scripts/chat.py
"""

import sys
import json
import httpx
from pathlib import Path

BASE_URL = "http://localhost:8000"


def health_check() -> dict:
    try:
        r = httpx.get(f"{BASE_URL}/api/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def query(text: str) -> str:
    try:
        r = httpx.post(
            f"{BASE_URL}/api/query",
            json={"query": text},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("answer", str(data))
    except httpx.HTTPStatusError as e:
        return f"[HTTP Error {e.response.status_code}] {e.response.text}"
    except Exception as e:
        return f"[Error] {e}"


def print_header():
    print("\n" + "═" * 60)
    print("  🤖  Rahul's RAG Voicebot — Terminal Chat")
    print("═" * 60)
    print("  Type your question and press Enter.")
    print("  Commands: 'health'  'quit' / 'exit' / Ctrl+C")
    print("═" * 60)


def main():
    print_header()

    # Health check
    h = health_check()
    if "error" in h:
        print(f"\n  ❌ Server not reachable: {h['error']}")
        print("  Make sure the API server is running:")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000\n")
        sys.exit(1)

    api_ok = h.get("api", False)
    qdrant_ok = h.get("qdrant", False)
    llm_ok = h.get("llm", False)
    status = "healthy" if all([api_ok, qdrant_ok, llm_ok]) else "degraded"
    status_icon = "✅" if status == "healthy" else "⚠️"
    print(f"\n  {status_icon} Server status: {status}")
    print(f"     API: {'✅' if api_ok else '❌'}  Qdrant: {'✅' if qdrant_ok else '❌'}  LLM: {'✅' if llm_ok else '❌'}")
    print()

    while True:
        try:
            user_input = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  👋 Goodbye!\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\n  👋 Goodbye!\n")
            break

        if user_input.lower() == "health":
            h = health_check()
            print(f"  ℹ️  Health: {json.dumps(h, indent=2)}\n")
            continue

        print("  🔍 Searching knowledge base...", end="\r")
        answer = query(user_input)
        print(" " * 40, end="\r")  # clear the searching line
        print(f"\n  🤖 Rahul's Bot:\n  {answer}\n")
        print("  " + "─" * 56)


if __name__ == "__main__":
    main()
