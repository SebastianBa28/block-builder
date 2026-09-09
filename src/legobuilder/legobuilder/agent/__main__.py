"""Interactive REPL entry point: python -m legobuilder.agent"""

import asyncio
import os
import sys

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from .agent import agent

APP_NAME = "lego_assembly_designer"


async def main():
    if not (os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")):
        print(
            "Error: Set GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY "
            "environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    session_service = InMemorySessionService()
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name=APP_NAME, user_id="local_user",
    )

    print("Magnetic Block Assembly Designer Agent")
    print("Type your request (Ctrl+C to quit):\n")

    try:
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break
            if not user_input.strip():
                continue

            message = types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            )
            for event in runner.run(
                user_id="local_user",
                session_id=session.id,
                new_message=message,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(f"Agent: {part.text}")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
