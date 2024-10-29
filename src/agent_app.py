# Sung-Cheol Kim Copyright 2024, All rights reserved.

import os
import instructor
from openai import OpenAI
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


def main():
    settings = Settings()

    console = Console()
    memory = AgentMemory()
    initial_message = BaseAgentOutputSchema(chat_message="Hello, How can I assist you today?")
    memory.add_message("assistant", initial_message)

    client = instructor.from_openai(OpenAI(api_key=settings.OPENAI_API_KEY))

    agent = BaseAgent(
        config=BaseAgentConfig(client=client, model="gpt-4o-mini", memory=memory),
    )

    default_system_prompt = agent.system_prompt_generator.generate_prompt()
    console.print(Panel(default_system_prompt, width=console.width, style="bold cyan"), style="bold cyan")
    console.print(Text("Agent:", style="bold green"), end=" ")
    console.print(Text(initial_message.chat_message, style="bold green"))

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ")
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Exiting chat...")
            break

        input_schema = BaseAgentInputSchema(chat_message=user_input)
        response = agent.run(input_schema)

        agent_message = Text(response.chat_message, style="bold green")
        console.print(Text("Agent:", style="bold green"), end=" ")
        console.print(agent_message)


if __name__ == "__main__":
    main()
