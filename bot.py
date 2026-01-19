#!/usr/bin/env python3
"""Telegram bot with custom Opus agent loop for quick queries and logging."""

import asyncio
import base64
import fcntl
import hashlib
import json
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import litellm

from session import (
    Session,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    Attachment,
    get_session as get_session_store,
    clear_session as clear_session_store,
    get_all_sessions,
)
from adapters import (
    ModelAdapter,
    get_reasoning_adapter,
    get_vision_adapter,
    get_long_context_adapter,
    get_search_adapter,
)
from prompts import (
    IMAGE_DESCRIBE,
    IMAGE_DESCRIBE_WITH_CAPTION,
    AUDIO_TRANSCRIBE,
    PDF_SUMMARIZE,
    SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
)
from e2b_sandbox import sandbox_manager
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

load_dotenv()

# Config
TOKEN = os.environ["TG_BOT_TOKEN"]
ALLOWED_USERS = {int(x) for x in os.environ.get("ALLOWED_USERS", "").split(",") if x}
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / "life"))
ATTACHMENTS_DIR = DATA_DIR / "attachments"
WAKEUPS_FILE = DATA_DIR / "wakeups.json"

# Derived paths are functions so tests can patch DATA_DIR


def get_stream_file() -> Path:
    return DATA_DIR / "stream.txt"


def get_stream_lock() -> Path:
    return DATA_DIR / ".stream.txt.lock"


# User identity mapping: user_id -> display name
# Format in env: "123456:Matthew,789012:Daisy"
# First user in ALLOWED_USERS is treated as owner
USER_NAMES = {0: "Scheduled"}  # 0 = scheduled wakeup (system)
for entry in os.environ.get("USER_NAMES", "").split(","):
    if ":" in entry:
        uid, name = entry.split(":", 1)
        USER_NAMES[int(uid.strip())] = name.strip()

# Owner is first allowed user
OWNER_ID = next(iter(ALLOWED_USERS)) if ALLOWED_USERS else None
OWNER_NAME = USER_NAMES.get(OWNER_ID, "User") if OWNER_ID else "User"

# Session config
SESSION_TIMEOUT = 60 * 60  # 1 hour
MAX_TOOL_ITERATIONS = 50

# Stream config (derived from DATA_DIR)
# (kept as constants for runtime use; tests may patch DATA_DIR, so stream tools
# should prefer get_stream_file()/get_stream_lock())
STREAM_FILE = get_stream_file()
STREAM_LOCK = get_stream_lock()

# Global bot reference for tools that need to send messages
_bot = None

# Tool definitions for Opus
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "stream_tail",
            "description": "Read the last n lines of stream.txt. MUST call at the start of every new session to load recent context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of lines to read from the end (default 50)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stream_range",
            "description": "Read a specific range of lines from stream.txt (1-indexed, inclusive). Use with stream_timeline to navigate to specific dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_line": {
                        "type": "integer",
                        "description": "Start line number (1-indexed, inclusive)",
                    },
                    "to_line": {
                        "type": "integer",
                        "description": "End line number (1-indexed, inclusive)",
                    },
                },
                "required": ["from_line", "to_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stream_append",
            "description": "Append raw text to the end of stream.txt. You control formatting - include date headers (# YYYY-MM-DD), newlines, tags as needed. MUST call after every response to log the interaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw text to append. Include newlines (\\n) for multiple lines.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stream_replace",
            "description": "Replace complete line(s) in the last 50 lines of stream.txt. The from_text must match one or more complete lines exactly once. Partial line matches are rejected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_text": {
                        "type": "string",
                        "description": "The exact complete line(s) to replace (must match full lines, unique within last 50 lines)",
                    },
                    "to_text": {
                        "type": "string",
                        "description": "The replacement text",
                    },
                },
                "required": ["from_text", "to_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stream_timeline",
            "description": "Get line ranges for each date header (# YYYY-MM-DD) in stream.txt. Use to navigate to specific dates with stream_range.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_stream",
            "description": "Ask a question about the entire stream.txt file. Uses a long-context model to search/analyze the full file. Use for queries like 'when did I last...', 'have I ever...', 'find all mentions of...', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask about the stream history",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stream_find",
            "description": "Find relevant sections in stream.txt. Returns (date, line_range, reason) tuples. Use this first to identify relevant sections, then stream_range to read specific ones. More efficient than ask_stream when you need the actual content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the stream history",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "session_brief",
            "description": "Get a comprehensive session brief including: tail 50 lines, timeline, and AI-generated summary of urgent items, calendar events, active todos, upcoming events, and retrieval aids. Use at start of sessions or when user asks for a briefing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for weather, news, facts, recommendations, prices, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random_pick",
            "description": "Pick n random items from a list without replacement. Useful for random selection, shuffling, or sampling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of items to pick from",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of items to pick (defaults to 1)",
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_wakeup",
            "description": "Schedule a future reminder or notification. The bot will wake up at the specified time, run the prompt (with full agent capabilities), and send the result to the user. Use for reminders, follow-ups, time-based notifications, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wake_time": {
                        "type": "string",
                        "description": "When to wake up. ISO datetime (e.g. '2026-01-15T09:00') or relative time (e.g. 'in 30 minutes', 'in 2 hours', 'tomorrow 9am')",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What to do when waking up. This prompt will be run through the agent with full tool access. E.g. 'Remind owner about the 9am meeting' or 'Check the weather and send morning briefing'",
                    },
                    "recurring": {
                        "type": "string",
                        "description": "Optional recurrence pattern: 'daily', 'weekly', 'hourly', or null for one-time",
                    },
                },
                "required": ["wake_time", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_wakeups",
            "description": "List all scheduled wakeups/reminders for this chat.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_wakeup",
            "description": "Cancel a scheduled wakeup by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wakeup_id": {
                        "type": "string",
                        "description": "The ID of the wakeup to cancel (from list_wakeups)",
                    },
                },
                "required": ["wakeup_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_attachment",
            "description": "Ask a question about a previously sent attachment (image, voice, audio, PDF). Use when you need details not covered in the initial description, want to re-examine content, or need specific information extracted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "description": "The attachment ID from [Image/Voice/Audio/PDF ID: ...] in the conversation",
                    },
                    "question": {
                        "type": "string",
                        "description": "What you want to know about the attachment",
                    },
                },
                "required": ["attachment_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_attachment",
            "description": "Send an attachment back to the user. Can send images, audio files, PDFs, or any file from the attachments folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "description": "The attachment ID (from previous messages or memory search)",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption to include with the attachment",
                    },
                },
                "required": ["attachment_id"],
            },
        },
    },
    # --- E2B Sandbox Tools ---
    {
        "type": "function",
        "function": {
            "name": "e2b_upload",
            "description": "Upload an attachment to the E2B sandbox for processing. Use when you need to run code on a file (xlsx, video, etc). Returns the remote path in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "description": "The attachment ID to upload",
                    },
                },
                "required": ["attachment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "e2b_run",
            "description": "Run a shell command in the E2B sandbox. Returns stdout/stderr. Working dir: /home/user/workspace. Pre-installed: Python 3.11, Node.js, pip, npm, apt, standard linux utils. Most packages need installing first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (e.g., 'python script.py', 'pip install pandas')",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "e2b_read",
            "description": "Read a text file from the E2B sandbox. Use to inspect code output, logs, or generated text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file (relative to workspace or absolute)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "e2b_ask_file",
            "description": "Send a file from the E2B sandbox to vision model for analysis. Use for audio transcription, image analysis, or any multimodal query on sandbox files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file in sandbox (relative to workspace or absolute)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Question or instruction for Gemini about the file",
                    },
                },
                "required": ["path", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "e2b_download",
            "description": "Download a file from the E2B sandbox and save as a local attachment. Returns the new attachment_id for use with send_attachment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file in sandbox (relative to workspace or absolute)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_task",
            "description": "Run a browser automation task using HyperAgent. Without session_id: runs in ephemeral session that is created and destroyed with the task (no login state, no timeout concerns). With session_id: runs on persistent session (from browser_session) that preserves login state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language description of what to do in the browser",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional: ID of persistent session (from browser_session). If omitted, runs in fresh ephemeral session.",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_session",
            "description": "Create a persistent browser session (10 min timeout) for manual user login. User receives live URL via Telegram to login manually. Returns session_id to pass to browser_task. Use only when task requires authentication that can't be automated - ask user to confirm when login is complete.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def get_system_prompt() -> str:
    """Get the system prompt with owner name substituted."""
    return SYSTEM_PROMPT.format(owner_name=OWNER_NAME)


def get_session(chat_id: int) -> Session:
    """Get or create session for chat, handling timeout.

    This is a compatibility wrapper around the session module.
    """
    return get_session_store(chat_id, timeout=SESSION_TIMEOUT)


def clear_session(chat_id: int) -> None:
    """Clear session for chat."""
    clear_session_store(chat_id)


def hash_messages(messages: list[Message]) -> str:
    """Hash a list of messages for prefix consistency verification."""

    # Convert semantic messages to a hashable representation
    def msg_to_dict(m: Message) -> dict:
        return {
            "role": m.role.value,
            "content": m.content,
            "tool_calls": [(tc.name, tc.arguments, tc.call_id) for tc in m.tool_calls],
            "tool_results": [(tr.call_id, tr.content) for tr in m.tool_results],
        }

    serialized = json.dumps(
        [msg_to_dict(m) for m in messages], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


# --- Attachment storage ---


def save_attachment_from_bytes(data: bytes, ext: str) -> tuple[str, Path]:
    """Save attachment bytes directly to permanent storage. Returns (attachment_id, path)."""
    ATTACHMENTS_DIR.mkdir(exist_ok=True)

    content_hash = hashlib.sha256(data).hexdigest()[:6]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    attachment_id = f"{timestamp}-{content_hash}"

    dest_path = ATTACHMENTS_DIR / f"{attachment_id}{ext}"
    dest_path.write_bytes(data)

    print(f"[attachment] Saved {len(data)} bytes -> {dest_path}", flush=True)
    return attachment_id, dest_path


def get_attachment_path(attachment_id: str) -> Path | None:
    """Get the path to an attachment file by ID (with or without extension)."""
    # Try exact match first (if extension included)
    exact = ATTACHMENTS_DIR / attachment_id
    if exact.exists() and exact.is_file():
        return exact

    # Strip extension if present and try glob
    stem = Path(attachment_id).stem
    matches = list(ATTACHMENTS_DIR.glob(f"{stem}.*"))
    # Filter out .json metadata files (legacy)
    matches = [m for m in matches if m.suffix != ".json"]
    return matches[0] if matches else None


def get_attachment_type(file_path: Path) -> str:
    """Infer attachment type from file extension."""
    ext = file_path.suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        return "image"
    elif ext in (".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac"):
        return "audio"
    elif ext in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
        return "video"
    elif ext == ".pdf":
        return "pdf"
    return "unknown"


def get_mime_type(file_path: Path) -> str:
    """Get MIME type from file extension."""
    ext = file_path.suffix.lower()
    mime_map = {
        # Images
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        # Audio
        ".ogg": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        # Video
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        # PDF
        ".pdf": "application/pdf",
    }
    return mime_map.get(ext, "application/octet-stream")


# --- Tool implementations ---


# --- Stream tool implementations ---

# Backwards-compat aliases (tests and older prompts)


def tool_read_stream_tail(n: int = 50) -> str:
    return tool_stream_tail(n)


def _read_stream_lines() -> list[str]:
    """Read stream.txt and return lines. Auto-creates if doesn't exist."""
    stream_file = get_stream_file()
    if not stream_file.exists():
        today = datetime.now().strftime("%Y-%m-%d")
        stream_file.write_text(f"# {today}\n")
    return stream_file.read_text().splitlines()


def _format_lines(lines: list[str], start_line: int, total: int) -> str:
    """Format lines with line numbers (1-indexed) and header."""
    end_line = start_line + len(lines) - 1
    header = f"Lines {start_line}-{end_line} (total: {total}):"
    numbered = [f"{start_line + i}\t{line}" for i, line in enumerate(lines)]
    return header + "\n" + "\n".join(numbered)


def tool_stream_tail(n: int = 50) -> str:
    """Read last n lines of stream.txt."""
    if n <= 0:
        return "Error: n must be positive."

    lines = _read_stream_lines()
    total = len(lines)

    if total == 0:
        return "Lines 0-0 (total: 0):\n(empty file)"

    tail_lines = lines[-n:] if len(lines) > n else lines
    start_line = total - len(tail_lines) + 1  # 1-indexed

    return _format_lines(tail_lines, start_line, total)


def tool_stream_range(from_line: int, to_line: int) -> str:
    """Read a specific range of lines (1-indexed, inclusive)."""
    lines = _read_stream_lines()
    total = len(lines)

    if total == 0:
        return "Lines 0-0 (total: 0):\n(empty file)"

    # Swap if from > to
    if from_line > to_line:
        from_line, to_line = to_line, from_line

    # Store original request for error message
    orig_from, orig_to = from_line, to_line

    # Clamp to valid range
    from_line = max(1, from_line)
    to_line = min(total, to_line)

    if from_line > total:
        return f"Lines {orig_from}-{orig_from} (requested {orig_from}-{orig_to}, total: {total}):\n(out of range)"

    # Convert to 0-indexed for slicing
    selected = lines[from_line - 1 : to_line]

    if not selected:
        return f"Lines {from_line}-{to_line} (total: {total}):\n(no lines in range)"

    return _format_lines(selected, from_line, total)


def tool_stream_append(text: str) -> str:
    """Append raw text to stream.txt with file locking."""
    if not text:
        return "Error: text cannot be empty."

    stream_file = get_stream_file()
    stream_lock = get_stream_lock()

    with open(stream_lock, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if stream_file.exists():
                content = stream_file.read_text()
            else:
                today = datetime.now().strftime("%Y-%m-%d")
                content = f"# {today}\n"

            # Append: strip trailing newline from original, add newline if text doesn't start with one
            content = content.rstrip("\n")
            if not text.startswith("\n"):
                content += "\n"
            content += text.rstrip("\n") + "\n"  # Ensure exactly one trailing newline
            stream_file.write_text(content)

            # Return context: last n+5 lines where n is lines added
            lines = content.splitlines()
            total = len(lines)
            lines_added = text.count("\n") + 1
            tail_count = lines_added + 5
            tail_lines = lines[-tail_count:] if len(lines) > tail_count else lines
            start_line = total - len(tail_lines) + 1

        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    return f"Appended {len(text)} chars. " + _format_lines(
        tail_lines, start_line, total
    )


def tool_stream_replace(from_text: str, to_text: str) -> str:
    """Replace text in the last 100 lines of stream.txt. Must match complete lines."""
    stream_file = get_stream_file()
    stream_lock = get_stream_lock()

    if not stream_file.exists():
        return "Error: stream.txt does not exist."

    # Validate from_text - must be non-empty
    from_text_stripped = from_text.strip("\n")
    if not from_text_stripped:
        return "Error: from_text cannot be empty."

    with open(stream_lock, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            content = stream_file.read_text()
            lines = content.split("\n")

            # Get last 100 lines window
            window_size = 100
            window_start = max(0, len(lines) - window_size)
            window_lines = lines[window_start:]

            # Validate from_text matches complete lines
            # Split from_text into lines and check each exists as a complete line
            from_lines = from_text.strip("\n").split("\n")

            # Find where the sequence of lines starts in the window
            match_indices = []
            for i in range(len(window_lines) - len(from_lines) + 1):
                if window_lines[i : i + len(from_lines)] == from_lines:
                    match_indices.append(i)

            if len(match_indices) == 0:
                return f"Error: from_text not found in last {window_size} lines."
            if len(match_indices) > 1:
                return f"Error: from_text found {len(match_indices)} times in last {window_size} lines - be more specific."

            match_idx = match_indices[0]

            # Perform replacement
            to_lines = to_text.strip("\n").split("\n") if to_text.strip("\n") else []
            new_window_lines = (
                window_lines[:match_idx]
                + to_lines
                + window_lines[match_idx + len(from_lines) :]
            )
            new_lines = lines[:window_start] + new_window_lines
            new_content = "\n".join(new_lines)

            stream_file.write_text(new_content)

            # Calculate context window: 5 lines before and after the change
            total_after = len(new_lines)
            change_start_line = window_start + match_idx + 1  # 1-indexed
            lines_in_replacement = len(to_lines) if to_lines else 0

            # Context: 5 before, the change, 5 after
            context_start = max(1, change_start_line - 5)
            context_end = min(
                total_after, change_start_line + max(lines_in_replacement - 1, 0) + 5
            )

            context_lines = new_lines[context_start - 1 : context_end]

        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    return f"Replaced. " + _format_lines(context_lines, context_start, total_after)


def tool_stream_timeline() -> str:
    """Get line ranges for each date header (# YYYY-MM-DD) in stream.txt."""
    lines = _read_stream_lines()
    total = len(lines)

    if total == 0:
        return "(empty file)"

    # Find all date headers and their line numbers
    date_pattern = re.compile(r"^# (\d{4}-\d{2}-\d{2})")
    dates = []  # [(line_num, date_str), ...]

    for i, line in enumerate(lines):
        match = date_pattern.match(line)
        if match:
            dates.append((i + 1, match.group(1)))  # 1-indexed

    if not dates:
        return f"No date headers found (total: {total} lines)"

    # Build output with line ranges
    output = []
    for i, (line_num, date_str) in enumerate(dates):
        if i + 1 < len(dates):
            end_line = dates[i + 1][0] - 1
        else:
            end_line = total
        output.append(f"# {date_str}: lines {line_num}-{end_line}")

    return "\n".join(output)


def _build_stream_content_message() -> tuple[Message, int] | None:
    """Build the cacheable stream content message. Returns (message, total_lines) or None if empty."""
    lines = _read_stream_lines()
    total = len(lines)

    if total == 0:
        return None

    # Numbered content - same format for both ask and find
    numbered_content = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
    content_msg = Message(
        role=MessageRole.USER,
        content=f"Stream file ({total} lines):\n{numbered_content}",
    )
    return content_msg, total


async def tool_ask_stream(query: str) -> str:
    """Ask a question about the entire stream.txt file using long-context model."""
    result = _build_stream_content_message()
    if result is None:
        return "Stream is empty."

    content_msg, _ = result

    task_msg = Message(
        role=MessageRole.USER,
        content=f"""Answer this question about the stream file above. Be concise and cite specific line numbers when relevant. If the information is not in the file, say so.

Question: {query}""",
    )

    try:
        adapter = get_long_context_adapter()
        response, _ = await adapter.complete(
            system_prompt="",
            messages=[content_msg, task_msg],
        )
        return response.content or ""
    except asyncio.TimeoutError:
        return "Ask stream timed out"
    except Exception as e:
        return f"Ask stream error: {e}"


async def tool_stream_find(query: str) -> str:
    """Find relevant sections in stream.txt, returning (date, line_range, reason) tuples."""
    result = _build_stream_content_message()
    if result is None:
        return "Stream is empty."

    content_msg, _ = result

    task_msg = Message(
        role=MessageRole.USER,
        content=f"""Find all sections in the stream file above that are relevant to the query. Return a list of matches, one per line.

Format: YYYY-MM-DD:[start-end] reason
Example output:
2025-01-15:[142-158] discussed project X timeline and blockers
2025-01-18:[201-215] follow-up on project X with new estimates

Rules:
- Include line ranges that contain relevant information
- Expand ranges to include full context (don't cut mid-conversation)
- Keep reason brief (under 10 words) - just enough context to decide if worth reading
- If nothing relevant found, return: (no matches)
- Return ONLY the formatted lines, no preamble or explanation

Query: {query}""",
    )

    try:
        adapter = get_long_context_adapter()
        response, _ = await adapter.complete(
            system_prompt="",
            messages=[content_msg, task_msg],
        )
        return response.content or "(no response)"
    except asyncio.TimeoutError:
        return "Stream find timed out"
    except Exception as e:
        return f"Stream find error: {e}"


def _find_latest_snapshot() -> Path | None:
    """Find the most recent snapshot-YYYY-MM-DD-HH-MM.txt file."""
    pattern = re.compile(r"^snapshot-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})\.txt$")
    snapshots = []

    for f in DATA_DIR.iterdir():
        match = pattern.match(f.name)
        if match:
            snapshots.append((match.group(1), f))

    if not snapshots:
        return None

    # Sort by timestamp string (lexicographic works for this format)
    snapshots.sort(key=lambda x: x[0], reverse=True)
    return snapshots[0][1]


def _get_date_from_snapshot_filename(path: Path) -> str | None:
    """Extract date (YYYY-MM-DD) from snapshot-YYYY-MM-DD-HH-MM.txt filename."""
    match = re.search(r"snapshot-(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}\.txt$", path.name)
    return match.group(1) if match else None


def _extract_stream_from_date(lines: list[str], from_date: str) -> list[str]:
    """Extract stream entries from given date onwards (inclusive)."""
    result = []
    include = False

    for line in lines:
        # Check for date header
        date_match = re.match(r"^# (\d{4}-\d{2}-\d{2})$", line)
        if date_match:
            line_date = date_match.group(1)
            include = line_date >= from_date

        if include:
            result.append(line)

    return result


def tool_session_brief() -> str:
    """Get session brief: snapshot + stream from snapshot date onwards."""
    lines = _read_stream_lines()
    total = len(lines)

    if total == 0:
        return "Stream is empty."

    # Check for snapshot
    snapshot_path = _find_latest_snapshot()
    snapshot_date: str | None = None

    if snapshot_path:
        snapshot_date = _get_date_from_snapshot_filename(snapshot_path)

    if snapshot_path and snapshot_date:
        # Snapshot exists: return snapshot + stream from snapshot date onwards
        snapshot_content = snapshot_path.read_text()

        # Get all date headers in the stream
        date_pattern = re.compile(r"^# (\d{4}-\d{2}-\d{2})$")
        dates_in_stream = []
        for line in lines:
            match = date_pattern.match(line)
            if match:
                dates_in_stream.append(match.group(1))

        # Start from snapshot date, expand backwards until we have ≥50 lines
        from_date = snapshot_date
        delta_lines = _extract_stream_from_date(lines, from_date)

        # Find dates before snapshot_date to expand if needed
        earlier_dates = sorted(
            [d for d in dates_in_stream if d < snapshot_date], reverse=True
        )

        while len(delta_lines) < 50 and earlier_dates:
            from_date = earlier_dates.pop(0)
            delta_lines = _extract_stream_from_date(lines, from_date)

        if delta_lines:
            stream_start = total - len(delta_lines) + 1
            stream_output = "\n".join(
                f"{stream_start + i}: {line}" for i, line in enumerate(delta_lines)
            )
            stream_header = (
                f"=== STREAM FROM {from_date} ({len(delta_lines)} lines) ==="
            )
        else:
            stream_output = "(no entries)"
            stream_header = "=== STREAM ==="

        return f"""=== SNAPSHOT ({snapshot_path.name}) ===
{snapshot_content}

{stream_header}
{stream_output}"""
    else:
        # No snapshot: return full stream
        stream_output = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

        return f"""=== NO SNAPSHOT - FULL STREAM ({total} lines) ===
{stream_output}"""


async def tool_web_search(query: str) -> str:
    """Web search using Grok online via OpenRouter."""
    try:
        adapter = get_search_adapter()
        msg = Message(role=MessageRole.USER, content=query)
        response, _ = await adapter.complete(
            system_prompt="",
            messages=[msg],
            extra_body={"plugins": [{"id": "web"}]},
        )
        return response.content or ""
    except asyncio.TimeoutError:
        return "Web search timed out"
    except Exception as e:
        return f"Search error: {e}"


def tool_random_pick(items: list[str], n: int = 1) -> str:
    """Pick n random items from a list without replacement."""
    if not items:
        return "Error: Empty list provided"
    if n < 1:
        return "Error: n must be at least 1"
    if n > len(items):
        n = len(items)  # Cap at list size instead of erroring

    picked = random.sample(items, n)
    if n == 1:
        return picked[0]
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(picked))


# --- Wakeup/Scheduler tools ---


WAKEUPS_LOCK_FILE = Path(__file__).parent / ".wakeups.lock"


def load_wakeups() -> list[dict]:
    """Load wakeups from JSON file."""
    if not WAKEUPS_FILE.exists():
        return []
    try:
        return json.loads(WAKEUPS_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return []


def save_wakeups(wakeups: list[dict]) -> None:
    """Save wakeups to JSON file."""
    WAKEUPS_FILE.write_text(json.dumps(wakeups, indent=2))


def parse_wake_time(wake_time_str: str) -> datetime | None:
    """Parse wake time from various formats."""
    now = datetime.now()
    original = wake_time_str.strip()
    wake_time_str = original.lower()

    # Try ISO format first (use original case for parsing)
    for fmt in [
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(original, fmt)
        except ValueError:
            continue

    # Relative time patterns
    if wake_time_str.startswith("in "):
        rest = wake_time_str[3:]
        # "in 30 minutes", "in 2 hours", "in 1 hour"
        match = re.match(
            r"(\d+)\s*(minute|min|minutes|mins|hour|hours|hr|hrs|day|days)", rest
        )
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            if unit in ("minute", "min", "minutes", "mins"):
                return now + timedelta(minutes=num)
            elif unit in ("hour", "hours", "hr", "hrs"):
                return now + timedelta(hours=num)
            elif unit in ("day", "days"):
                return now + timedelta(days=num)

    # "tomorrow 9am", "tomorrow 14:00"
    if wake_time_str.startswith("tomorrow"):
        rest = wake_time_str[8:].strip()
        tomorrow = now + timedelta(days=1)
        # Parse time part
        time_match = re.match(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", rest)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            ampm = time_match.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # Default to 9am tomorrow
        return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)

    # Try just time today/tomorrow: "9am", "14:00", "9:30pm"
    time_match = re.match(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?$", wake_time_str)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        ampm = time_match.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # If time has passed today, schedule for tomorrow
        if target <= now:
            target += timedelta(days=1)
        return target

    return None


def tool_schedule_wakeup(
    wake_time: str, prompt: str, recurring: str | None, chat_id: int
) -> str:
    """Schedule a wakeup."""
    parsed_time = parse_wake_time(wake_time)
    if not parsed_time:
        return f"Could not parse wake time: '{wake_time}'. Try formats like '2026-01-15T09:00', 'in 30 minutes', 'tomorrow 9am', or '14:00'"

    if parsed_time <= datetime.now():
        return f"Wake time {parsed_time} is in the past. Please specify a future time."

    # Validate recurring
    valid_recurring = {None, "daily", "weekly", "hourly"}
    if recurring and recurring not in valid_recurring:
        return f"Invalid recurring value: '{recurring}'. Use 'daily', 'weekly', 'hourly', or omit for one-time."

    wakeup_id = str(uuid.uuid4())[:8]
    wakeup = {
        "id": wakeup_id,
        "wake_time": parsed_time.isoformat(),
        "prompt": prompt,
        "chat_id": chat_id,
        "recurring": recurring,
        "created_at": datetime.now().isoformat(),
    }

    with open(WAKEUPS_LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            wakeups = load_wakeups()
            wakeups.append(wakeup)
            save_wakeups(wakeups)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    recur_msg = f" (recurring: {recurring})" if recurring else " (one-time)"
    return f"Scheduled wakeup {wakeup_id} for {parsed_time.strftime('%Y-%m-%d %H:%M')}{recur_msg}\nPrompt: {prompt}"


def tool_list_wakeups(chat_id: int) -> str:
    """List wakeups for a chat."""
    wakeups = load_wakeups()
    chat_wakeups = [w for w in wakeups if w.get("chat_id") == chat_id]

    if not chat_wakeups:
        return "No scheduled wakeups."

    lines = ["Scheduled wakeups:"]
    for w in sorted(chat_wakeups, key=lambda x: x["wake_time"]):
        recur = f" [{w['recurring']}]" if w.get("recurring") else ""
        lines.append(f"  {w['id']}: {w['wake_time']}{recur} - {w['prompt'][:50]}...")

    return "\n".join(lines)


def tool_cancel_wakeup(wakeup_id: str, chat_id: int) -> str:
    """Cancel a wakeup."""
    with open(WAKEUPS_LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            wakeups = load_wakeups()
            original_count = len(wakeups)

            # Only allow canceling own wakeups
            wakeups = [
                w
                for w in wakeups
                if not (w.get("id") == wakeup_id and w.get("chat_id") == chat_id)
            ]

            if len(wakeups) == original_count:
                return f"Wakeup {wakeup_id} not found or doesn't belong to this chat."

            save_wakeups(wakeups)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    return f"Cancelled wakeup {wakeup_id}"


# --- Attachment tools ---


async def tool_ask_attachment(attachment_id: str, question: str) -> str:
    """Ask a question about an attachment using Gemini."""
    # Find the attachment file
    attachment_path = get_attachment_path(attachment_id)
    if not attachment_path:
        return f"Error: Attachment {attachment_id} not found"

    try:
        file_bytes = attachment_path.read_bytes()
        b64 = base64.b64encode(file_bytes).decode()
        mime_type = get_mime_type(attachment_path)

        # Build message with attachment
        attachment = Attachment(
            attachment_id=attachment_id,
            mime_type=mime_type,
            data_b64=b64,
        )
        msg = Message(
            role=MessageRole.USER,
            content=question,
            attachments=[attachment],
        )

        adapter = get_vision_adapter()
        response, _ = await adapter.complete(
            system_prompt="",
            messages=[msg],
        )
        return response.content if response.content else "No response from analysis"

    except Exception as e:
        print(f"[ask_attachment] Error: {e}", flush=True)
        return f"Error analyzing attachment: {e}"


async def tool_send_attachment(
    attachment_id: str, chat_id: int, caption: str | None = None
) -> str:
    """Send an attachment to the user."""
    global _bot
    if _bot is None:
        return "Error: Bot not initialized"

    # Find the attachment file
    attachment_path = get_attachment_path(attachment_id)
    if not attachment_path:
        return f"Error: Attachment {attachment_id} not found"

    attachment_type = get_attachment_type(attachment_path)

    try:
        with open(attachment_path, "rb") as f:
            if attachment_type == "image":
                await _bot.send_photo(chat_id=chat_id, photo=f, caption=caption)
            elif attachment_type in ("voice", "audio"):
                await _bot.send_audio(chat_id=chat_id, audio=f, caption=caption)
            elif attachment_type == "pdf":
                await _bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    caption=caption,
                    filename=f"{attachment_id}.pdf",
                )
            else:
                # Generic document
                await _bot.send_document(chat_id=chat_id, document=f, caption=caption)

        return f"Sent {attachment_type} {attachment_id}"

    except Exception as e:
        print(f"[send_attachment] Error: {e}", flush=True)
        return f"Error sending attachment: {e}"


# --- E2B Sandbox tools ---


async def tool_e2b_upload(attachment_id: str, chat_id: int) -> str:
    """Upload an attachment to the E2B sandbox."""
    # Find the attachment file
    attachment_path = get_attachment_path(attachment_id)
    if not attachment_path:
        return f"Error: Attachment {attachment_id} not found"

    try:
        content = attachment_path.read_bytes()
        filename = attachment_path.name
        remote_path, is_new = await sandbox_manager.upload_file(
            chat_id, filename, content
        )
        result = f"Uploaded to sandbox: {remote_path} ({len(content)} bytes)"
        if is_new:
            result = (
                "(New sandbox created - previous files/packages cleared)\n" + result
            )
        return result
    except Exception as e:
        print(f"[e2b_upload] Error: {e}", flush=True)
        return f"Error uploading to sandbox: {e}"


async def tool_e2b_run(command: str, chat_id: int) -> str:
    """Run a command in the E2B sandbox."""
    try:
        result = await sandbox_manager.run_command(chat_id, command)
        output = ""
        if result["is_new_sandbox"]:
            output = "(New sandbox created - previous files/packages cleared)\n"
        if result["stdout"]:
            output += result["stdout"]
        if result["stderr"]:
            if result["stdout"]:
                output += "\n--- STDERR ---\n"
            output += result["stderr"]
        if not result["success"]:
            output += f"\n[Exit code: {result['exit_code']}]"
        return output.strip() or "(No output)"
    except Exception as e:
        print(f"[e2b_run] Error: {e}", flush=True)
        return f"Error running command: {e}"


async def tool_e2b_read(path: str, chat_id: int) -> str:
    """Read a text file from the E2B sandbox."""
    try:
        content, error = await sandbox_manager.read_file(chat_id, path)
        if error:
            return f"Error: {error}"
        return content or "(Empty file)"
    except Exception as e:
        print(f"[e2b_read] Error: {e}", flush=True)
        return f"Error reading file: {e}"


async def tool_e2b_ask_file(path: str, query: str, chat_id: int) -> str:
    """Send a sandbox file to vision model for analysis."""
    try:
        # Download file from sandbox
        content, error = await sandbox_manager.download_file(chat_id, path)
        if error:
            return f"Error: {error}"
        if not content:
            return "Error: Empty file"

        # Determine MIME type from extension (reuse get_mime_type)
        mime_type = get_mime_type(Path(path))

        # Encode and build attachment
        b64 = base64.b64encode(content).decode()
        attachment = Attachment(
            attachment_id=f"sandbox:{path}",
            mime_type=mime_type,
            data_b64=b64,
        )
        msg = Message(
            role=MessageRole.USER,
            content=query,
            attachments=[attachment],
        )

        adapter = get_vision_adapter()
        response, _ = await adapter.complete(
            system_prompt="",
            messages=[msg],
            timeout=180,  # Longer timeout for large files
        )
        return response.content if response.content else "No response from vision model"

    except Exception as e:
        print(f"[e2b_ask_file] Error: {e}", flush=True)
        return f"Error: {e}"


async def tool_e2b_download(path: str, chat_id: int) -> str:
    """Download a file from E2B sandbox and save as local attachment."""
    try:
        # Download from sandbox
        content, error = await sandbox_manager.download_file(chat_id, path)
        if error:
            return f"Error: {error}"
        if not content:
            return "Error: Empty file"

        # Determine extension
        ext = Path(path).suffix.lower() or ".bin"

        # Save as attachment
        attachment_id, attachment_path = save_attachment_from_bytes(content, ext)
        return f"Downloaded and saved as attachment: {attachment_id} ({len(content)} bytes)"

    except Exception as e:
        print(f"[e2b_download] Error: {e}", flush=True)
        return f"Error downloading file: {e}"


async def tool_browser_session(chat_id: int) -> str:
    """Create a persistent browser session for manual user login.

    Returns session_id to agent, sends live URL to user.
    Session lasts 10 minutes.
    """
    from hyperbrowser import AsyncHyperbrowser
    from hyperbrowser.models import CreateSessionParams, ScreenConfig

    api_key = os.environ.get("HYPERBROWSER_API_KEY")
    if not api_key:
        return "Error: HYPERBROWSER_API_KEY not set."

    try:
        async with AsyncHyperbrowser(api_key=api_key) as client:
            print("[browser_session] Creating persistent session...", flush=True)
            session = await client.sessions.create(
                CreateSessionParams(
                    screen=ScreenConfig(width=1280, height=720),
                    timeout_minutes=10,
                )
            )
            session_id = session.id
            live_url = session.live_url
            print(f"[browser_session] Session created: {session_id}", flush=True)

            # Send live URL to user
            if _bot and live_url:
                try:
                    await _bot.send_message(
                        chat_id=chat_id,
                        text=f"Browser session for manual login (10 min): {live_url}",
                    )
                except Exception as e:
                    print(f"[browser_session] Failed to send live URL: {e}", flush=True)

            return f"Session created: {session_id}\nUser has been sent the live URL for manual login. Ask user to confirm when login is complete, then use browser_task with this session_id."

    except Exception as e:
        print(f"[browser_session] Error: {e}", flush=True)
        return f"Error: {e}"


async def tool_browser_task(task: str, session_id: str | None = None) -> str:
    """Run a browser automation task using HyperAgent.

    If session_id is None, runs in ephemeral session (API-managed, ends with task).
    If session_id is provided, runs on that persistent session.
    """
    from hyperbrowser import AsyncHyperbrowser
    from hyperbrowser.models import StartHyperAgentTaskParams

    api_key = os.environ.get("HYPERBROWSER_API_KEY")
    if not api_key:
        return "Error: HYPERBROWSER_API_KEY not set. For read-only web content, use web_search instead."

    try:
        async with AsyncHyperbrowser(api_key=api_key) as client:
            if session_id:
                print(
                    f"[browser_task] Running on session {session_id}: {task[:100]}...",
                    flush=True,
                )
            else:
                print(
                    f"[browser_task] Running ephemeral task: {task[:100]}...",
                    flush=True,
                )

            result = await client.agents.hyper_agent.start_and_wait(
                StartHyperAgentTaskParams(
                    task=task,
                    version="1.1.0",
                    llm="claude-sonnet-4-5",
                    max_steps=30,
                    session_id=session_id,
                    keep_browser_open=bool(session_id),
                )
            )

            print(f"[browser_task] Status: {result.status}", flush=True)

            if result.status == "completed" and result.data:
                final_result = result.data.final_result or "No result returned"
                return f"Task completed.\n\nResult:\n{final_result}"
            else:
                error = result.error or "Unknown error"
                return f"Task {result.status}.\n\nError: {error}"

    except Exception as e:
        print(f"[browser_task] Error: {e}", flush=True)
        return f"Error: {e}"


async def execute_tool(name: str, args: dict, chat_id: int) -> str:
    """Execute a tool and return result."""
    # Async tools
    if name == "web_search":
        return await tool_web_search(args["query"])
    if name == "ask_stream":
        return await tool_ask_stream(args["query"])
    if name == "stream_find":
        return await tool_stream_find(args["query"])
    if name == "session_brief":
        return tool_session_brief()
    if name == "ask_attachment":
        return await tool_ask_attachment(args["attachment_id"], args["question"])
    if name == "send_attachment":
        return await tool_send_attachment(
            args["attachment_id"], chat_id, args.get("caption")
        )
    # Browser automation
    if name == "browser_session":
        return await tool_browser_session(chat_id)
    if name == "browser_task":
        return await tool_browser_task(args["task"], args.get("session_id"))
    # E2B sandbox tools
    if name == "e2b_upload":
        return await tool_e2b_upload(args["attachment_id"], chat_id)
    if name == "e2b_run":
        return await tool_e2b_run(args["command"], chat_id)
    if name == "e2b_read":
        return await tool_e2b_read(args["path"], chat_id)
    if name == "e2b_ask_file":
        return await tool_e2b_ask_file(args["path"], args["query"], chat_id)
    if name == "e2b_download":
        return await tool_e2b_download(args["path"], chat_id)

    # Sync tools - dispatch table
    dispatch = {
        "stream_tail": (tool_stream_tail, lambda a, c: (a.get("n", 50),)),
        "read_stream_tail": (tool_read_stream_tail, lambda a, c: (a.get("n", 50),)),
        "stream_range": (
            tool_stream_range,
            lambda a, c: (a["from_line"], a["to_line"]),
        ),
        "stream_append": (tool_stream_append, lambda a, c: (a["text"],)),
        "stream_replace": (
            tool_stream_replace,
            lambda a, c: (a["from_text"], a["to_text"]),
        ),
        "stream_timeline": (tool_stream_timeline, lambda a, c: ()),
        "random_pick": (tool_random_pick, lambda a, c: (a["items"], a.get("n", 1))),
        "schedule_wakeup": (
            tool_schedule_wakeup,
            lambda a, c: (a["wake_time"], a["prompt"], a.get("recurring"), c),
        ),
        "list_wakeups": (tool_list_wakeups, lambda a, c: (c,)),
        "cancel_wakeup": (tool_cancel_wakeup, lambda a, c: (a["wakeup_id"], c)),
    }

    if name not in dispatch:
        return f"Unknown tool: {name}"

    handler, args_extractor = dispatch[name]
    # Run sync tools in executor to allow true parallelism
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, handler, *args_extractor(args, chat_id))


# --- Preprocessing ---


async def preprocess_image(file_path: str, caption: str = "") -> str | None:
    """Describe image using Gemini - detailed description."""
    if caption:
        prompt = IMAGE_DESCRIBE_WITH_CAPTION.format(caption=caption)
    else:
        prompt = IMAGE_DESCRIBE

    try:
        print(f"[preprocess_image] Processing {file_path}", flush=True)
        image_bytes = Path(file_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        mime_type = get_mime_type(Path(file_path))

        attachment = Attachment(
            attachment_id="preprocess",
            mime_type=mime_type,
            data_b64=b64,
        )
        msg = Message(
            role=MessageRole.USER,
            content=prompt,
            attachments=[attachment],
        )

        adapter = get_vision_adapter()
        response, _ = await adapter.complete(
            system_prompt="",
            messages=[msg],
            timeout=60,
        )
        result = response.content
        if result:
            print(f"[preprocess_image] Success: {result[:50]}...", flush=True)
            return result
        print("[preprocess_image] Failed: empty response", flush=True)
    except Exception as e:
        print(f"[preprocess_image] Exception: {e}", flush=True)
    return None


async def preprocess_audio(file_path: str) -> str | None:
    """Transcribe audio using Gemini - full transcription."""
    try:
        print(f"[preprocess_audio] Transcribing {file_path}", flush=True)
        audio_bytes = Path(file_path).read_bytes()
        b64 = base64.b64encode(audio_bytes).decode()
        mime_type = get_mime_type(Path(file_path))

        attachment = Attachment(
            attachment_id="preprocess",
            mime_type=mime_type,
            data_b64=b64,
        )
        msg = Message(
            role=MessageRole.USER,
            content=AUDIO_TRANSCRIBE,
            attachments=[attachment],
        )

        # Load session brief for context (names, jargon, terminology)
        system_prompt = ""
        try:
            session_brief = tool_session_brief()
            system_prompt = f"Context about the speaker (use for recognizing names, jargon, terminology):\n\n{session_brief}"
            print(
                f"[preprocess_audio] Loaded session brief context ({len(session_brief)} chars)",
                flush=True,
            )
        except Exception as e:
            print(f"[preprocess_audio] Failed to load session brief: {e}", flush=True)

        adapter = get_vision_adapter()
        response, _ = await adapter.complete(
            system_prompt=system_prompt,
            messages=[msg],
        )
        result = response.content
        if result:
            print(f"[preprocess_audio] Success: {result[:50]}...", flush=True)
            return result
        print("[preprocess_audio] Failed: empty response", flush=True)
    except Exception as e:
        print(f"[preprocess_audio] Exception: {e}", flush=True)
    return None


async def preprocess_pdf(file_path: str) -> str | None:
    """Summarize PDF using Gemini - brief summary."""
    try:
        print(f"[preprocess_pdf] Processing {file_path}", flush=True)
        pdf_bytes = Path(file_path).read_bytes()
        b64 = base64.b64encode(pdf_bytes).decode()

        attachment = Attachment(
            attachment_id="preprocess",
            mime_type="application/pdf",
            data_b64=b64,
        )
        msg = Message(
            role=MessageRole.USER,
            content=PDF_SUMMARIZE,
            attachments=[attachment],
        )

        adapter = get_vision_adapter()
        response, _ = await adapter.complete(
            system_prompt="",
            messages=[msg],
            timeout=120,
        )
        result = response.content
        if result:
            print(f"[preprocess_pdf] Success: {result[:50]}...", flush=True)
            return result
        print("[preprocess_pdf] Failed: empty response", flush=True)
    except Exception as e:
        print(f"[preprocess_pdf] Exception: {e}", flush=True)
    return None


# --- Agent loop ---


async def run_agent(
    user_message: str,
    chat_id: int,
    adapter: ModelAdapter | None = None,
    session: Session | None = None,
) -> str | None:
    """Run the agent loop with tools.

    Message batching: If new messages arrive while running, they are queued
    in session.pending_messages and injected as additional user messages.
    LLM responses (text or tool requests) are disposable and can be discarded
    when new messages arrive. Tool results are NOT disposable - once tools
    execute, their results must be sent to the LLM.

    Args:
        user_message: The user's message to process.
        chat_id: The chat ID for session management.
        adapter: Optional model adapter (defaults to reasoning adapter).
        session: Optional session override (for subagents with isolated context).

    Returns:
        The final response text, or None if no response.
    """
    # Default to main reasoning adapter
    if adapter is None:
        adapter = get_reasoning_adapter()

    # Get session (or use provided one for subagents)
    if session is None:
        session = get_session(chat_id)

    system_prompt = get_system_prompt()

    # Add user message to session
    user_msg = Message(role=MessageRole.USER, content=user_message)
    session.messages.append(user_msg)

    # Hash the history prefix for consistency verification
    history_len = len(session.messages) - 1  # Exclude the message we just added
    history_hash = (
        hash_messages(session.messages[:history_len]) if history_len > 0 else "empty"
    )
    print(
        f"[run_agent] chat_id={chat_id}, model={adapter.model_name}, history={history_len}, history_hash={history_hash}",
        flush=True,
    )

    def drain_pending() -> bool:
        """Drain pending messages into session. Returns True if any were added."""
        if not session.pending_messages:
            return False
        pending = session.pending_messages[:]
        session.pending_messages.clear()
        print(f"[run_agent] Draining {len(pending)} pending messages", flush=True)
        for text in pending:
            session.messages.append(Message(role=MessageRole.USER, content=text))
        return True

    response_msg: Message | None = None

    for iteration in range(MAX_TOOL_ITERATIONS):
        # Drain any pending messages before API call
        drain_pending()

        print(
            f"[run_agent] API call iteration={iteration}, messages={len(session.messages)}",
            flush=True,
        )

        # Retry logic for transient API errors
        max_retries = 3
        retry_delay = 2  # seconds, will double each retry
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                response_msg, usage = await adapter.complete(
                    system_prompt=system_prompt,
                    messages=session.messages,
                    tools=TOOLS,
                )
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                # Retry on transient errors
                is_transient = any(
                    x in error_msg
                    for x in [
                        "overloaded",
                        "rate",
                        "internal server",
                        "503",
                        "429",
                        "timeout",
                        "connection",
                        "ratelimit",
                        "service unavailable",
                    ]
                )
                if is_transient and attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    print(
                        f"[run_agent] Transient API error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}",
                        flush=True,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                # Non-retryable or max retries exhausted
                print(f"[run_agent] API error: {e}", flush=True)
                return f"API error: {e}"
        else:
            # All retries exhausted
            print(
                f"[run_agent] API error after {max_retries} retries: {last_error}",
                flush=True,
            )
            return f"API error (after {max_retries} retries): {last_error}"

        # Log usage stats
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read", 0)
            cache_write = usage.get("cache_write", 0)
            print(
                f"[run_agent] tokens: in={input_tokens} out={output_tokens} cache_write={cache_write} cache_read={cache_read}",
                flush=True,
            )

        # Check if done (no tool calls)
        if not response_msg.tool_calls:
            # Check for pending messages one more time
            if drain_pending():
                # New messages arrived - discard this response, re-call API
                continue

            # Truly done - save to session and return
            session.messages.append(response_msg)

            print(
                f"[run_agent] Done. Total messages: {len(session.messages)}",
                flush=True,
            )

            return response_msg.content

        # Has tool calls - but check pending first!
        # Tool call requests are disposable (not yet executed), so if user sent
        # new messages, discard this response and let model see the new context
        if drain_pending():
            # New messages arrived - discard tool call request, re-call API
            print(
                "[run_agent] Discarding tool call request due to pending messages",
                flush=True,
            )
            continue

        # No pending - safe to execute tools
        # Send intermediate text to user immediately (if present)
        if response_msg.content:
            chunks = (
                [
                    response_msg.content[i : i + 4096]
                    for i in range(0, len(response_msg.content), 4096)
                ]
                if len(response_msg.content) > 4096
                else [response_msg.content]
            )
            for chunk in chunks:
                try:
                    await _bot.send_message(
                        chat_id=chat_id, text=chunk, parse_mode="HTML"
                    )
                except TelegramError:
                    await _bot.send_message(chat_id=chat_id, text=chunk)

        # Commit: append assistant message with tool calls
        session.messages.append(response_msg)

        # Execute ALL tool calls in parallel - NOT interruptible
        # (tools may have side effects, API requires all tool_calls to have matching results)
        async def run_tool(tc: ToolCall) -> ToolResult:
            result = await execute_tool(tc.name, tc.arguments, chat_id)
            return ToolResult(call_id=tc.call_id, content=result)

        tool_results = await asyncio.gather(
            *[run_tool(tc) for tc in response_msg.tool_calls]
        )

        # Append tool results as a single message
        result_msg = Message(
            role=MessageRole.TOOL_RESULT,
            tool_results=list(tool_results),
        )
        session.messages.append(result_msg)

        # Loop continues - tool results now committed, will be sent to model

    # Max iterations reached
    print(
        f"[run_agent] Max iterations reached. Total messages: {len(session.messages)}",
        flush=True,
    )

    # Display-only warning (not saved to session)
    last_content = response_msg.content if response_msg else ""
    if last_content:
        return f"{last_content}\n\n(iteration limit reached)"
    return "(iteration limit reached)"


# --- Telegram handlers ---


def is_allowed(update: Update) -> bool:
    if not ALLOWED_USERS:
        return True
    user = update.effective_user
    allowed = user.id in ALLOWED_USERS
    if not allowed:
        print(
            f"[REJECTED] user_id={user.id}, username={user.username}, name={user.first_name} {user.last_name}",
            flush=True,
        )
    return allowed


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    await update.message.reply_text(
        "Hey! Send me a message, photo, or voice note.\n/new - start fresh conversation"
    )


async def new_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    clear_session(update.effective_chat.id)
    await update.message.reply_text("Conversation cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return

    msg = update.message
    text = msg.text or msg.caption or ""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    # Helper to download, preprocess, persist, and format attachment
    async def process_attachment(
        file_id: str,
        ext: str,
        attachment_type: str,
        preprocess_fn,
        original_filename: str | None = None,
    ) -> str | None:
        """Download to memory, save to attachments, preprocess, return formatted string."""
        # Download to memory
        file = await context.bot.get_file(file_id)
        data = bytes(await file.download_as_bytearray())

        # Save directly to attachments folder (hash in memory, no temp file)
        attachment_id, attachment_path = save_attachment_from_bytes(data, ext)

        # Preprocess from saved file
        if attachment_type == "image":
            description = await preprocess_fn(str(attachment_path), msg.caption or "")
        else:
            description = await preprocess_fn(str(attachment_path))

        if not description:
            # Preprocessing failed - clean up the saved file
            try:
                attachment_path.unlink()
            except Exception:
                pass
            return None

        # Format message based on type
        # Note: attachment_id is the ONLY way to retrieve the file later - filename is not stored
        notice = f"(Save ID '{attachment_id}' to memory to retrieve this file in future sessions)"
        if attachment_type == "voice":
            return f'[Voice {attachment_id}: "{description}"]\n{notice}'
        elif attachment_type == "audio":
            fname = f" {original_filename}" if original_filename else ""
            return f'[Audio {attachment_id}{fname}: "{description}"]\n{notice}'
        elif attachment_type == "image":
            fname = f" {original_filename}" if original_filename else ""
            return f"[Image {attachment_id}{fname}: {description}]\n{notice}"
        elif attachment_type == "pdf":
            fname = f" {original_filename}" if original_filename else ""
            return f"[PDF {attachment_id}{fname}: {description}]\n{notice}"
        else:
            return f"[Attachment {attachment_id}: {description}]\n{notice}"

    # Handle voice messages
    if msg.voice:
        try:
            result = await process_attachment(
                msg.voice.file_id, ".ogg", "voice", preprocess_audio
            )
            if result:
                text = f"{result}\n\n{text}".strip()
            else:
                text = f"[Voice message received but couldn't transcribe]\n\n{text}".strip()
        except Exception as e:
            print(f"[handle_message] Voice download/process failed: {e}", flush=True)
            text = f"[Voice message received but download failed]\n\n{text}".strip()

    # Handle photos (sent as photos, not documents)
    if msg.photo:
        try:
            photo = msg.photo[-1]  # Highest resolution
            result = await process_attachment(
                photo.file_id, ".jpg", "image", preprocess_image
            )
            if result:
                text = f"{result}\n\n{text}".strip()
            else:
                text = f"[Image received but couldn't process]\n\n{text}".strip()
        except Exception as e:
            print(f"[handle_message] Photo download/process failed: {e}", flush=True)
            text = f"[Image received but download failed]\n\n{text}".strip()

    # Handle documents (images, PDFs, audio files)
    if msg.document:
        doc = msg.document
        mime = doc.mime_type or ""
        original_filename = doc.file_name
        ext = Path(original_filename).suffix.lower() if original_filename else ""

        if mime.startswith("image/"):
            # Image sent as document
            ext = ext or ".jpg"
            try:
                result = await process_attachment(
                    doc.file_id, ext, "image", preprocess_image, original_filename
                )
                if result:
                    text = f"{result}\n\n{text}".strip()
                else:
                    text = f"[Image received but couldn't process]\n\n{text}".strip()
            except Exception as e:
                print(
                    f"[handle_message] Image doc download/process failed: {e}",
                    flush=True,
                )
                text = f"[Image received but download failed: {original_filename}]\n\n{text}".strip()

        elif mime == "application/pdf" or ext == ".pdf":
            # PDF document
            try:
                result = await process_attachment(
                    doc.file_id, ".pdf", "pdf", preprocess_pdf, original_filename
                )
                if result:
                    text = f"{result}\n\n{text}".strip()
                else:
                    text = f"[PDF received but couldn't process: {original_filename}]\n\n{text}".strip()
            except Exception as e:
                print(f"[handle_message] PDF download/process failed: {e}", flush=True)
                text = f"[PDF received but download failed: {original_filename}]\n\n{text}".strip()

        elif mime.startswith("audio/") or ext in (
            ".mp3",
            ".wav",
            ".m4a",
            ".aac",
            ".flac",
            ".ogg",
        ):
            # Audio file
            ext = ext or ".mp3"
            try:
                result = await process_attachment(
                    doc.file_id, ext, "audio", preprocess_audio, original_filename
                )
                if result:
                    text = f"{result}\n\n{text}".strip()
                else:
                    text = f"[Audio received but couldn't transcribe: {original_filename}]\n\n{text}".strip()
            except Exception as e:
                print(
                    f"[handle_message] Audio download/process failed: {e}", flush=True
                )
                text = f"[Audio received but download failed: {original_filename}]\n\n{text}".strip()

        else:
            # Any other document type - save without preprocessing
            # Model can use e2b_upload + e2b_run to inspect these
            ext = ext or ".bin"
            try:
                file = await context.bot.get_file(doc.file_id)
                data = bytes(await file.download_as_bytearray())
                attachment_id, _ = save_attachment_from_bytes(data, ext)
                notice = f"(Save ID '{attachment_id}' to memory. Use e2b_upload to inspect with code.)"
                text = f"[File {attachment_id} {original_filename} ({mime}, {len(data)} bytes)]\n{notice}\n\n{text}".strip()
            except Exception as e:
                print(f"[handle_message] File download failed: {e}", flush=True)
                text = f"[File received but download failed: {original_filename}]\n\n{text}".strip()

    if not text:
        return

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    session = get_session(chat_id)

    # Inject time and user identity into the user message
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
    user_name = USER_NAMES.get(user_id, "Unknown") if user_id else "Unknown"
    formatted_text = f"[{current_time}] [{user_name}] {text}"

    # If agent is running, queue message and return immediately (don't wait)
    if session.lock.locked():
        session.pending_messages.append(formatted_text)
        print(
            f"[handle_message] Queued message (pending={len(session.pending_messages)})",
            flush=True,
        )
        return

    async with session.lock:
        response = await run_agent(
            formatted_text,
            chat_id,
        )

        # No content - nothing to send
        if not response:
            return

        # Send response (split if too long)
        chunks = (
            [response[i : i + 4096] for i in range(0, len(response), 4096)]
            if len(response) > 4096
            else [response]
        )

        for chunk in chunks:
            try:
                await msg.reply_text(chunk, parse_mode="HTML")
            except TelegramError as e:
                print(f"[handle_message] HTML parse failed: {e}", flush=True)
                # Fallback to plain text
                await msg.reply_text(chunk)


# --- Scheduler ---

SCHEDULER_INTERVAL = 60  # Check every 60 seconds


async def process_wakeup(wakeup: dict, bot) -> None:
    """Process a single wakeup - run agent and send message."""
    chat_id = wakeup["chat_id"]
    prompt = wakeup["prompt"]
    wakeup_id = wakeup["id"]

    print(f"[scheduler] Processing wakeup {wakeup_id}: {prompt[:50]}...", flush=True)

    try:
        # Run agent with the wakeup prompt (include current datetime for context)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        response = await run_agent(
            f"[{current_time}] [SCHEDULED WAKEUP] {prompt}",
            chat_id,
        )

        # Send the response
        if response:
            chunks = (
                [response[i : i + 4096] for i in range(0, len(response), 4096)]
                if len(response) > 4096
                else [response]
            )

            for chunk in chunks:
                try:
                    await bot.send_message(
                        chat_id=chat_id, text=chunk, parse_mode="HTML"
                    )
                except TelegramError as e:
                    print(f"[scheduler] HTML parse failed: {e}", flush=True)
                    await bot.send_message(chat_id=chat_id, text=chunk)
        else:
            print(f"[scheduler] Wakeup {wakeup_id} returned no response", flush=True)

        print(f"[scheduler] Completed wakeup {wakeup_id}", flush=True)

    except Exception as e:
        print(f"[scheduler] Error processing wakeup {wakeup_id}: {e}", flush=True)
        # Try to notify user of failure
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"Scheduled reminder failed: {e}\n\nOriginal prompt: {prompt[:200]}",
            )
        except Exception:
            pass


async def scheduler_loop(app) -> None:
    """Background task that checks and processes wakeups."""
    bot = app.bot
    print("[scheduler] Scheduler started", flush=True)

    while True:
        try:
            now = datetime.now()
            wakeups = load_wakeups()
            remaining = []
            to_process = []

            for w in wakeups:
                wake_time = datetime.fromisoformat(w["wake_time"])
                if wake_time <= now:
                    to_process.append(w)
                else:
                    remaining.append(w)

            # Process due wakeups
            for w in to_process:
                await process_wakeup(w, bot)

                # Handle recurring
                if w.get("recurring"):
                    recur = w["recurring"]
                    wake_time = datetime.fromisoformat(w["wake_time"])

                    if recur == "hourly":
                        next_time = wake_time + timedelta(hours=1)
                    elif recur == "daily":
                        next_time = wake_time + timedelta(days=1)
                    elif recur == "weekly":
                        next_time = wake_time + timedelta(weeks=1)
                    else:
                        next_time = None

                    if next_time:
                        # Skip past times (in case bot was down)
                        while next_time <= now:
                            if recur == "hourly":
                                next_time += timedelta(hours=1)
                            elif recur == "daily":
                                next_time += timedelta(days=1)
                            elif recur == "weekly":
                                next_time += timedelta(weeks=1)

                        w["wake_time"] = next_time.isoformat()
                        remaining.append(w)
                        print(
                            f"[scheduler] Rescheduled {w['id']} for {next_time}",
                            flush=True,
                        )

            # Save updated wakeups
            if to_process:
                save_wakeups(remaining)

            # Check for expired sessions
            now_ts = time.time()
            expired_chats = []
            for chat_id, session in list(get_all_sessions().items()):
                if now_ts - session.last_used > SESSION_TIMEOUT:
                    expired_chats.append(chat_id)

            for chat_id in expired_chats:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="Session cleared after 1 hour of inactivity.",
                    )
                except Exception as e:
                    print(
                        f"[scheduler] Failed to notify session timeout for {chat_id}: {e}",
                        flush=True,
                    )
                clear_session(chat_id)

        except Exception as e:
            print(f"[scheduler] Error in scheduler loop: {e}", flush=True)

        await asyncio.sleep(SCHEDULER_INTERVAL)


_scheduler_task: asyncio.Task | None = None


async def post_init(app) -> None:
    """Called after the application is initialized."""
    global _bot, _scheduler_task
    _bot = app.bot
    # Start the scheduler as a background task
    _scheduler_task = asyncio.create_task(scheduler_loop(app))


async def post_shutdown(app) -> None:
    """Called after the application is shut down - cleanup resources."""
    global _scheduler_task

    # Cancel the scheduler task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        print("[shutdown] Scheduler task cancelled", flush=True)

    # Close litellm's async HTTP clients to avoid the warning:
    # "RuntimeWarning: coroutine 'close_litellm_async_clients' was never awaited"
    try:
        await litellm.aclient_session_cleanup()
    except Exception as e:
        print(f"[shutdown] Error cleaning up litellm clients: {e}", flush=True)


def main() -> None:
    # Ensure attachments directory exists
    ATTACHMENTS_DIR.mkdir(exist_ok=True)

    # Use HTTPXRequest without proxy for Telegram API (proxy is for Gemini geo-bypass only)
    # This avoids "TTL expired" errors when the SOCKS proxy has issues
    request = HTTPXRequest(proxy=None)

    app = (
        Application.builder()
        .token(TOKEN)
        .request(request)
        .get_updates_request(request)
        .concurrent_updates(True)  # Allow handlers to run concurrently
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_session))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))

    print(f"Bot starting... allowed users: {ALLOWED_USERS or 'all'}")
    print(f"Model: {get_reasoning_adapter().model_name}")
    print(f"Session timeout: {SESSION_TIMEOUT // 3600}h")
    print(f"Scheduler interval: {SCHEDULER_INTERVAL}s")
    print(f"Attachments dir: {ATTACHMENTS_DIR}")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
