"""
Unit tests for ChatSession.
Run with: pytest tests/test_chat.py -v
"""

import json
import pytest
from src.chat import ChatSession, ChatMessage


class TestChatSession:
    def test_add_user_message(self):
        session = ChatSession()
        session.add_user("Hello")
        assert len(session) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"

    def test_add_assistant_message(self):
        session = ChatSession()
        session.add_assistant("Hi there", sources=[{"source": "doc.pdf", "page": 1}])
        assert len(session) == 1
        assert session.messages[0].role == "assistant"
        assert len(session.messages[0].sources) == 1

    def test_multi_turn(self):
        session = ChatSession()
        session.add_user("Q1")
        session.add_assistant("A1")
        session.add_user("Q2")
        session.add_assistant("A2")
        assert len(session) == 4
        roles = [m.role for m in session.messages]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_clear(self):
        session = ChatSession()
        session.add_user("test")
        session.add_assistant("response")
        session.clear()
        assert len(session) == 0

    def test_get_history_for_llm(self):
        session = ChatSession()
        session.add_user("What is AI?")
        session.add_assistant("AI is artificial intelligence.")
        history = session.get_history_for_llm()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "What is AI?"}
        assert history[1]["role"] == "assistant"

    def test_export_json(self):
        session = ChatSession(session_id="test-123")
        session.add_user("Question?")
        session.add_assistant("Answer.")
        exported = json.loads(session.export_json())
        assert exported["session_id"] == "test-123"
        assert len(exported["messages"]) == 2
        assert "exported_at" in exported

    def test_export_markdown(self):
        session = ChatSession()
        session.add_user("What is RAG?")
        session.add_assistant("RAG is Retrieval-Augmented Generation.")
        md = session.export_markdown()
        assert "What is RAG?" in md
        assert "RAG is Retrieval" in md
        assert "**You**" in md
        assert "**Assistant**" in md

    def test_export_csv(self):
        session = ChatSession()
        session.add_user("Test")
        session.add_assistant("Response")
        csv_data = session.export_csv()
        assert "user" in csv_data
        assert "assistant" in csv_data
        assert "Test" in csv_data

    def test_sources_in_assistant(self):
        session = ChatSession()
        sources = [
            {"source": "a.pdf", "page": 1, "score": 0.9},
            {"source": "b.pdf", "page": 2, "score": 0.7},
        ]
        session.add_assistant("Answer with citations.", sources=sources)
        msg = session.messages[0]
        assert len(msg.sources) == 2
        assert msg.sources[0]["source"] == "a.pdf"
