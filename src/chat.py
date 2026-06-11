"""
Chat session management: history, export, multi-turn context handling.
"""

from __future__ import annotations

import json
import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.utils import now_iso


@dataclass
class ChatMessage:
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=now_iso)
    sources: list[dict] = field(default_factory=list)
    retrieval_mode: str = ""
    model: str = ""
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "sources": self.sources,
            "retrieval_mode": self.retrieval_mode,
            "model": self.model,
        }


class ChatSession:
    """
    Manages a single user conversation session.
    Keeps message history and supports multi-turn context passing.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id or f"session_{now_iso()}"
        self.messages: list[ChatMessage] = []
        self.created_at: str = now_iso()

    # ── History access ────────────────────────────────────────────────────────

    def add_message(self, msg: ChatMessage) -> None:
        self.messages.append(msg)

    def add_user(self, content: str) -> None:
        self.add_message(ChatMessage(role="user", content=content))

    def add_assistant(
        self,
        content: str,
        sources: list[dict] | None = None,
        model: str = "",
        retrieval_mode: str = "",
    ) -> None:
        self.add_message(ChatMessage(
            role="assistant",
            content=content,
            sources=sources or [],
            model=model,
            retrieval_mode=retrieval_mode,
        ))

    def get_history_for_llm(self) -> list[dict[str, str]]:
        """Return plain role/content dicts suitable for passing to the LLM."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)

    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self) -> str:
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "exported_at": now_iso(),
            "messages": [m.to_dict() for m in self.messages],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_markdown(self) -> str:
        lines = [
            f"# Chat Export — {self.session_id}",
            f"_Exported: {now_iso()}_\n",
        ]
        for msg in self.messages:
            role_label = "**You**" if msg.role == "user" else "**Assistant**"
            lines.append(f"### {role_label}  \n_{msg.timestamp}_\n")
            lines.append(msg.content)
            if msg.sources:
                lines.append("\n**Sources:**")
                for src in msg.sources:
                    lines.append(
                        f"- {src.get('source', '')} — "
                        f"Page {src.get('page', '?')} "
                        f"(score: {src.get('score', 0):.3f})"
                    )
            lines.append("\n---\n")
        return "\n".join(lines)

    def export_csv(self) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["timestamp", "role", "content", "sources", "model"],
        )
        writer.writeheader()
        for msg in self.messages:
            writer.writerow({
                "timestamp": msg.timestamp,
                "role": msg.role,
                "content": msg.content,
                "sources": "; ".join(
                    f"{s.get('source')} p.{s.get('page')}" for s in msg.sources
                ),
                "model": msg.model,
            })
        return output.getvalue()
