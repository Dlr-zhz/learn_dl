from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class LLMProvider:
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def _client(self):
        # Return None to trigger echo fallback if OpenAI SDK is unavailable
        # or if no API key is configured via explicit param or environment.
        if OpenAI is None:
            return None
        effective_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not effective_key:
            return None
        return OpenAI(api_key=effective_key, base_url=self.base_url)

    def complete(self, messages: List[Dict[str, str]]) -> str:
        client = self._client()
        if client is None:
            # Echo fallback for environments without OpenAI installed
            joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return f"[Echo LLM]\n{joined}\n[End]"

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # Fallback to echo on runtime errors
            joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return f"[Echo LLM - Fallback due to error: {exc}]\n{joined}\n[End]"
