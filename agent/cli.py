from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .renderer import TemplateRenderer
from .memory import ConversationMemory
from .llm import LLMProvider
from .chat import ChatAgent


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(config: Dict[str, Any]) -> ChatAgent:
    templates_dir = config.get("templates_dir", "templates")
    template_name = config.get("template_name", "reply.j2")

    renderer = TemplateRenderer(templates_dir=templates_dir)
    memory = ConversationMemory(
        max_messages=int(config.get("memory", {}).get("max_messages", 20)),
        system_prompt=config.get("system_prompt"),
    )

    llm_cfg = config.get("llm", {})
    llm = LLMProvider(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        api_key=os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
        base_url=llm_cfg.get("base_url"),
    )

    return ChatAgent(
        renderer=renderer,
        memory=memory,
        llm=llm,
        template_name=template_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Template-driven multi-turn chat agent")
    parser.add_argument("--config", default="config/agent.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent = build_agent(cfg)

    print("Agent ready. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        output = agent.ask(user_input)
        print(output)


if __name__ == "__main__":
    main()
