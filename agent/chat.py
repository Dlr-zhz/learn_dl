from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .renderer import TemplateRenderer
from .memory import ConversationMemory
from .llm import LLMProvider


@dataclass
class ChatAgent:
    renderer: TemplateRenderer
    memory: ConversationMemory
    llm: LLMProvider
    template_name: str

    def ask(self, user_input: str, extra_context: Dict[str, Any] | None = None) -> str:
        self.memory.add("user", user_input)
        messages = self.memory.snapshot()
        assistant_reply = self.llm.complete(messages)
        self.memory.add("assistant", assistant_reply)

        render_context: Dict[str, Any] = {
            "last_user": user_input,
            "assistant_reply": assistant_reply,
            "messages": messages,
        }
        if extra_context:
            render_context.update(extra_context)

        return self.renderer.render(self.template_name, render_context)
