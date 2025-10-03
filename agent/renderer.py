from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template


@dataclass
class TemplateRenderer:
    templates_dir: str

    def __post_init__(self) -> None:
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        template: Template = self.env.get_template(template_name)
        return template.render(**context)
