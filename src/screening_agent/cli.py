"""Interactive terminal client (M3 acceptance):

python -m screening_agent.cli --new
"""

from __future__ import annotations

import argparse
import logging

from screening_agent.engine import Conversation
from screening_agent.llm.client import LLMClient
from screening_agent.store import Store


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="screening-agent")
    parser.add_argument("--new", action="store_true", help="start a new conversation")
    args = parser.parse_args()

    if not args.new:
        parser.error("only --new is supported right now")

    store = Store()
    client = LLMClient()
    conversation = Conversation(store=store, client=client)

    print(f"[conversation {conversation.id}]")
    print(f"agent: {conversation.start()}")
    while not conversation.finished:
        try:
            candidate_message = input("you: ")
        except EOFError:
            print("\n[input closed before the conversation finished]")
            break
        reply = conversation.step(candidate_message)
        print(f"agent: {reply}")

    if conversation.finished:
        print(f"[outcome: {conversation.outcome.value}]")


if __name__ == "__main__":
    main()
