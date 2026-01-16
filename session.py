"""LLM-agnostic session store and message types.

This module provides the core data structures for representing conversations
in a provider-independent way. Adapters convert these semantic types to
provider-specific formats.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


@dataclass
class ToolCall:
    """A tool invocation request from the assistant.

    Attributes:
        name: Name of the tool to invoke.
        arguments: Arguments to pass to the tool.
        call_id: Unique identifier for matching with result.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class ToolResult:
    """Result from executing a tool.

    Attributes:
        call_id: ID matching the original ToolCall.
        content: String result from the tool.
    """

    call_id: str
    content: str


@dataclass
class Attachment:
    """Embedded multimodal content (image, audio, PDF, etc).

    Attributes:
        attachment_id: ID for retrieval from attachment storage.
        mime_type: MIME type of the content.
        data_b64: Base64-encoded content.
    """

    attachment_id: str
    mime_type: str
    data_b64: str


@dataclass
class Message:
    """LLM-agnostic message representation.

    This is the core semantic unit of conversation. Adapters convert
    these to provider-specific formats.

    Attributes:
        role: Role of the message sender.
        content: Text content of the message.
        tool_calls: List of tool invocations (for assistant messages).
        tool_results: List of tool results (for tool_result messages).
        attachments: Embedded multimodal content.
        thinking: Extended thinking content (Anthropic-specific, preserved for caching).
        metadata: Additional provider-specific metadata to preserve.
    """

    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    thinking: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Per-chat conversation session.

    Attributes:
        messages: Conversation history as semantic messages.
        last_used: Timestamp of last activity (for timeout).
        lock: Async lock for concurrent access.
        pending_messages: Queued user messages while agent is running.
    """

    messages: list[Message] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    pending_messages: list[str] = field(default_factory=list)


# Global session storage
_sessions: dict[int, Session] = {}


def get_session(chat_id: int, timeout: float = 3600) -> Session:
    """Get or create session for a chat, clearing if timed out.

    Args:
        chat_id: Unique chat identifier.
        timeout: Session timeout in seconds (default 1 hour).

    Returns:
        The session for this chat.
    """
    now = time.time()

    if chat_id in _sessions:
        session = _sessions[chat_id]
        if now - session.last_used > timeout:
            # Session expired, create new one
            del _sessions[chat_id]
        else:
            session.last_used = now
            return session

    # Create new session
    session = Session()
    _sessions[chat_id] = session
    return session


def clear_session(chat_id: int) -> None:
    """Clear session for a chat."""
    _sessions.pop(chat_id, None)


def get_all_sessions() -> dict[int, Session]:
    """Get all sessions (for scheduler/cleanup)."""
    return _sessions
