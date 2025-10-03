from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Tuple
from collections import deque


Message = Dict[str, str]  # {"role": "user|assistant|system", "content": str}


@dataclass
class ConversationMemory:
    max_messages: int = 20
    system_prompt: str | None = None
    _messages: Deque[Message] = field(default_factory=deque, init=False)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        while len(self._messages) > self.max_messages:
            self._messages.popleft()

    def snapshot(self) -> List[Message]:
        messages: List[Message] = list(self._messages)
        if self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}] + messages
        return messages

    def clear(self) -> None:
        self._messages.clear()
