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
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import litellm

# Force httpx transport for SOCKS proxy support (aiohttp doesn't handle SOCKS properly)
litellm.disable_aiohttp_transport = True

from models import REASONING, reasoning_complete, vision_complete, search_complete
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

# Config
TOKEN = os.environ["TG_BOT_TOKEN"]
ALLOWED_USERS = {int(x) for x in os.environ.get("ALLOWED_USERS", "").split(",") if x}
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / "life"))
AGENTS_MD = Path(__file__).parent / "AGENTS.md"
ATTACHMENTS_DIR = DATA_DIR / "attachments"
WAKEUPS_FILE = DATA_DIR / "wakeups.json"

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

# Rotation config
ROTATION_THRESHOLD = 100 * 1024  # 100kb
ROTATION_HOUR = 4  # 4am
_last_rotation_check: str | None = None

# Global bot reference for tools that need to send messages
_bot = None

# In-memory sessions: {chat_id: {"messages": [...], "last_used": timestamp}}
sessions: dict[int, dict] = {}


# Tool definitions for Opus
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_stream",
            "description": "Read the entire current stream.txt file. Use this to understand recent context, tasks, and history.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_stream_tail",
            "description": "Read the last n lines of stream.txt. Use for quick recent context without loading the full file.",
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
            "name": "append_stream",
            "description": "Append raw text to the end of stream.txt. Does NOT add any formatting - appends exactly what you provide. You must include date headers (## YYYY-MM-DD), newlines, etc. yourself. Can append multiple lines at once. Use plain text without bullet prefixes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw text to append verbatim. Include newlines (\\n) for multiple lines. Example: '## 2025-01-15\\nFirst item\\nSecond item'",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_stream",
            "description": "Search across all stream files (current and archived) for information. Use for historical queries like 'when did I last...', 'have I ever...', etc. Runs in parallel across all stream files.",
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
            "name": "stream_replace",
            "description": "Replace text in the last 100 lines of stream.txt. Use for correcting recent entries, updating status, or fixing typos. The from_text must be complete lines (at least 1 full line) and must match exactly once within the 100-line window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_text": {
                        "type": "string",
                        "description": "The exact text to replace (must be complete lines, unique within last 100 lines)",
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
            "name": "list_attachments",
            "description": "List attachments that have been sent. Filter by type or search descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Optional: filter by type - 'image', 'voice', 'audio', 'pdf'",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional: search term to match against filenames or descriptions",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 20)",
                    },
                },
                "required": [],
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
                        "description": "The attachment ID to send (from list_attachments or previous messages)",
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
    {
        "type": "function",
        "function": {
            "name": "rotate_stream",
            "description": "Rotate stream.txt (archive current and reinitialize) with dream time analysis. Use when user requests rotation. This archives the current stream, extracts open commitments, analyzes patterns across archived streams, and creates a fresh stream with findings.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def extract_system_prompt() -> str:
    """Extract system prompt from AGENTS.md (excluding Tools section)."""
    if not AGENTS_MD.exists():
        return f"You are a helpful assistant. Your owner is {OWNER_NAME}."

    content = AGENTS_MD.read_text()

    # Skip the Tools section (bot has its own tool implementations)
    lines = content.split("\n")
    result = []
    skip_until_next_h2 = False

    for line in lines:
        if line.startswith("## Tools (terminal)"):
            skip_until_next_h2 = True
            continue
        if line.startswith("## ") and skip_until_next_h2:
            skip_until_next_h2 = False
        if not skip_until_next_h2:
            result.append(line)

    base_prompt = "\n".join(result).strip()

    return f"""{base_prompt}

---
Environment: Telegram
Output format: HTML (required)

Your owner is {OWNER_NAME}. When logging items requested by someone other than the owner, always note who requested it (e.g. "TODO: Pick up groceries (someone added)")."""


def get_session(chat_id: int) -> list[dict]:
    """Get or create session for chat, handling timeout."""
    now = time.time()

    if chat_id in sessions:
        session = sessions[chat_id]
        if now - session["last_used"] > SESSION_TIMEOUT:
            # Expired
            del sessions[chat_id]
        else:
            session["last_used"] = now
            # Ensure lock and interrupt exist (for safety if session structure changes)
            session.setdefault("lock", asyncio.Lock())
            session.setdefault("interrupt", asyncio.Event())
            return session["messages"]

    # New session
    sessions[chat_id] = {
        "messages": [],
        "last_used": now,
        "lock": asyncio.Lock(),
        "interrupt": asyncio.Event(),
    }
    return sessions[chat_id]["messages"]


def build_assistant_msg(msg) -> dict:
    """Build assistant message dict preserving all API-relevant fields."""
    result = {"role": "assistant", "content": msg.content}

    if msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]

    # Preserve reasoning (OpenRouter format)
    provider_fields = getattr(msg, "provider_specific_fields", None) or {}
    if provider_fields.get("reasoning_details"):
        result["reasoning_details"] = provider_fields["reasoning_details"]

    return result


def hash_messages(messages: list[dict]) -> str:
    """Hash a list of messages for prefix consistency verification."""
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def add_message_to_session(chat_id: int, message: dict) -> None:
    """Add a message dict to session history verbatim for prefix consistency."""
    messages = get_session(chat_id)
    messages.append(message)


def clear_session(chat_id: int) -> None:
    """Clear session for chat."""
    if chat_id in sessions:
        del sessions[chat_id]


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


def save_attachment_metadata(
    attachment_id: str,
    attachment_type: str,
    description: str,
    chat_id: int,
    user_id: int,
    original_filename: str | None = None,
) -> None:
    """Save attachment metadata to sidecar JSON file."""
    # Find the attachment file to get extension
    matches = list(ATTACHMENTS_DIR.glob(f"{attachment_id}.*"))
    if not matches:
        print(f"[attachment] Warning: No file found for {attachment_id}", flush=True)
        return

    ext = matches[0].suffix
    metadata_path = ATTACHMENTS_DIR / f"{attachment_id}.json"

    metadata = {
        "id": attachment_id,
        "type": attachment_type,
        "extension": ext,
        "original_filename": original_filename,
        "chat_id": chat_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "description": description,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[attachment] Saved metadata for {attachment_id}", flush=True)


def load_attachment_metadata(attachment_id: str) -> dict | None:
    """Load attachment metadata from sidecar JSON file."""
    metadata_path = ATTACHMENTS_DIR / f"{attachment_id}.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def get_attachment_path(attachment_id: str) -> Path | None:
    """Get the path to an attachment file by ID."""
    matches = list(ATTACHMENTS_DIR.glob(f"{attachment_id}.*"))
    # Filter out .json metadata files
    matches = [m for m in matches if m.suffix != ".json"]
    return matches[0] if matches else None


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
        # PDF
        ".pdf": "application/pdf",
    }
    return mime_map.get(ext, "application/octet-stream")


# --- Tool implementations ---


def tool_read_stream() -> str:
    """Read entire stream.txt."""
    stream_file = DATA_DIR / "stream.txt"
    if not stream_file.exists():
        return "stream.txt does not exist yet."
    return stream_file.read_text()


def tool_read_stream_tail(n: int = 50) -> str:
    """Read last n lines of stream.txt."""
    stream_file = DATA_DIR / "stream.txt"
    if not stream_file.exists():
        return "stream.txt does not exist yet."
    lines = stream_file.read_text().splitlines()
    tail = lines[-n:] if len(lines) > n else lines
    return "\n".join(tail)


def tool_append_stream(text: str) -> str:
    """Append raw text to stream.txt. Uses file locking. Returns last n+5 lines after append."""
    stream_file = DATA_DIR / "stream.txt"
    lock_file = DATA_DIR / ".stream.txt.lock"

    with open(lock_file, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if stream_file.exists():
                content = stream_file.read_text()
            else:
                content = "# stream\n"

            # Just append raw text with a leading newline to separate
            # User request: simple append, trust model for spacing/newlines
            if text:
                content = content.rstrip("\n")
                if not text.startswith("\n"):
                    content += "\n"
                content += text
                stream_file.write_text(content)

            # Return last n+5 lines where n is lines added
            lines_added = text.count("\n") + 1
            all_lines = content.splitlines()
            tail_count = lines_added + 5
            tail_lines = (
                all_lines[-tail_count:] if len(all_lines) > tail_count else all_lines
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    return f"Appended {len(text)} chars. Context:\n" + "\n".join(tail_lines)


def tool_stream_replace(from_text: str, to_text: str) -> str:
    """Replace text in the last 100 lines of stream.txt."""
    stream_file = DATA_DIR / "stream.txt"
    lock_file = DATA_DIR / ".stream.txt.lock"

    if not stream_file.exists():
        return "Error: stream.txt does not exist."

    # Validate from_text has at least one full line
    from_text = from_text.strip("\n")
    if not from_text:
        return "Error: from_text cannot be empty."
    if "\n" not in from_text and len(from_text.strip()) < 3:
        return "Error: from_text must be at least one meaningful line."

    with open(lock_file, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            content = stream_file.read_text()
            lines = content.split("\n")

            # Get last 50 lines window
            window_start = max(0, len(lines) - 50)
            window_lines = lines[window_start:]
            window_text = "\n".join(window_lines)

            # Check for match in window
            match_count = window_text.count(from_text)

            if match_count == 0:
                return "Error: from_text not found in last 50 lines."
            if match_count > 1:
                return f"Error: from_text found {match_count} times in last 50 lines - be more specific."

            # Perform replacement only in window
            new_window_text = window_text.replace(from_text, to_text, 1)
            new_lines = lines[:window_start] + new_window_text.split("\n")
            new_content = "\n".join(new_lines)

            stream_file.write_text(new_content)

            # Return last 50 lines as context
            result_lines = new_content.split("\n")
            tail_count = min(50, len(result_lines))
            tail = result_lines[-tail_count:]

        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    return f"Replaced. Last {tail_count} lines:\n" + "\n".join(tail)


async def _ask_single_log(query: str, log_path: Path) -> tuple[str, str]:
    """Query a single log file with Gemini. Returns (filename, answer)."""
    try:
        content = log_path.read_text()
        response = await vision_complete(
            messages=[
                {
                    "role": "user",
                    "content": f"Answer this question based on the log file content. Be concise. If the information is not in the log, say 'Not found in this log.'\n\nLog content:\n{content}\n\nQuestion: {query}",
                }
            ],
        )
        return (log_path.name, response.choices[0].message.content.strip())
    except asyncio.TimeoutError:
        return (log_path.name, "Timeout querying this log")
    except Exception as e:
        return (log_path.name, f"Error: {e}")


def log_access(query: str, source: str = "tg") -> None:
    """Append query to ACCESS.log for dream-time analysis."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    access_log = DATA_DIR / "ACCESS.log"
    line = f'{timestamp} {source} "{query}"\n'
    with open(access_log, "a") as f:
        f.write(line)


async def tool_ask_stream(query: str) -> str:
    """Query all stream*.txt files in parallel."""
    log_access(query, "tg")
    stream_files = list(DATA_DIR.glob("stream*.txt"))
    if not stream_files:
        return "No stream files found."

    # Run queries in parallel with asyncio.gather
    results = await asyncio.gather(*[_ask_single_log(query, f) for f in stream_files])

    # Format results
    output = []
    for filename, answer in results:
        if "Not found" not in answer:
            output.append(f"**{filename}**: {answer}")

    if not output:
        return "No relevant information found in any stream files."

    return "\n\n".join(output)


async def tool_web_search(query: str) -> str:
    """Web search using Grok online via OpenRouter."""
    try:
        response = await search_complete(
            messages=[{"role": "user", "content": query}],
            extra_body={"plugins": [{"id": "web"}]},
        )
        return response.choices[0].message.content.strip()
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

    metadata = load_attachment_metadata(attachment_id)
    if not metadata:
        return f"Error: Metadata for {attachment_id} not found"

    attachment_type = metadata.get("type", "unknown")

    try:
        file_bytes = attachment_path.read_bytes()
        b64 = base64.b64encode(file_bytes).decode()
        mime_type = get_mime_type(attachment_path)

        # Build the appropriate message based on type
        if attachment_type == "image":
            content = [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
            ]
        elif attachment_type in ("voice", "audio"):
            content = [
                {"type": "text", "text": question},
                {
                    "type": "file",
                    "file": {"file_data": f"data:{mime_type};base64,{b64}"},
                },
            ]
        elif attachment_type == "pdf":
            content = [
                {"type": "text", "text": question},
                {
                    "type": "file",
                    "file": {"file_data": f"data:{mime_type};base64,{b64}"},
                },
            ]
        else:
            return f"Error: Unsupported attachment type: {attachment_type}"

        response = await vision_complete(
            messages=[{"role": "user", "content": content}],
        )
        result = response.choices[0].message.content.strip()
        return result if result else "No response from analysis"

    except Exception as e:
        print(f"[ask_attachment] Error: {e}", flush=True)
        return f"Error analyzing attachment: {e}"


def tool_list_attachments(
    attachment_type: str | None = None,
    query: str | None = None,
    limit: int = 20,
) -> str:
    """List attachments, optionally filtered by type or search query."""
    if not ATTACHMENTS_DIR.exists():
        return "No attachments found."

    # Find all metadata files
    metadata_files = sorted(
        ATTACHMENTS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # Most recent first
    )

    results = []
    for meta_path in metadata_files:
        if len(results) >= limit:
            break

        try:
            metadata = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, IOError):
            continue

        # Filter by type
        if attachment_type and metadata.get("type") != attachment_type:
            continue

        # Filter by query (search in description and filename)
        if query:
            query_lower = query.lower()
            description = (metadata.get("description") or "").lower()
            filename = (metadata.get("original_filename") or "").lower()
            if query_lower not in description and query_lower not in filename:
                continue

        # Format result
        att_id = metadata.get("id", "unknown")
        att_type = metadata.get("type", "unknown")
        filename = metadata.get("original_filename")
        timestamp = metadata.get("timestamp", "")[:10]  # Just date part
        description = metadata.get("description", "")[:100]

        if filename:
            results.append(
                f"{att_id} [{att_type}] {filename} ({timestamp}): {description}..."
            )
        else:
            results.append(f"{att_id} [{att_type}] ({timestamp}): {description}...")

    if not results:
        if attachment_type:
            return f"No {attachment_type} attachments found."
        elif query:
            return f"No attachments matching '{query}' found."
        return "No attachments found."

    header = f"Found {len(results)} attachment(s):\n"
    return header + "\n".join(results)


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

    metadata = load_attachment_metadata(attachment_id)
    attachment_type = metadata.get("type", "unknown") if metadata else "unknown"

    try:
        with open(attachment_path, "rb") as f:
            if attachment_type == "image":
                await _bot.send_photo(chat_id=chat_id, photo=f, caption=caption)
            elif attachment_type in ("voice", "audio"):
                await _bot.send_audio(chat_id=chat_id, audio=f, caption=caption)
            elif attachment_type == "pdf":
                filename = metadata.get("original_filename") if metadata else None
                await _bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    caption=caption,
                    filename=filename or f"{attachment_id}.pdf",
                )
            else:
                # Generic document
                await _bot.send_document(chat_id=chat_id, document=f, caption=caption)

        return f"Sent {attachment_type} {attachment_id}"

    except Exception as e:
        print(f"[send_attachment] Error: {e}", flush=True)
        return f"Error sending attachment: {e}"


async def execute_tool(name: str, args: dict, chat_id: int) -> str:
    """Execute a tool and return result."""
    # Async tools
    if name == "ask_stream":
        return await tool_ask_stream(args["query"])
    if name == "web_search":
        return await tool_web_search(args["query"])
    if name == "ask_attachment":
        return await tool_ask_attachment(args["attachment_id"], args["question"])
    if name == "send_attachment":
        return await tool_send_attachment(
            args["attachment_id"], chat_id, args.get("caption")
        )
    if name == "rotate_stream":
        from rotation import rotate_stream

        return await rotate_stream(force=True)

    # Sync tools - dispatch table
    dispatch = {
        "read_stream": (tool_read_stream, lambda a, c: ()),
        "read_stream_tail": (tool_read_stream_tail, lambda a, c: (a.get("n", 50),)),
        "append_stream": (tool_append_stream, lambda a, c: (a["text"],)),
        "stream_replace": (
            tool_stream_replace,
            lambda a, c: (a["from_text"], a["to_text"]),
        ),
        "random_pick": (tool_random_pick, lambda a, c: (a["items"], a.get("n", 1))),
        "schedule_wakeup": (
            tool_schedule_wakeup,
            lambda a, c: (a["wake_time"], a["prompt"], a.get("recurring"), c),
        ),
        "list_wakeups": (tool_list_wakeups, lambda a, c: (c,)),
        "cancel_wakeup": (tool_cancel_wakeup, lambda a, c: (a["wakeup_id"], c)),
        "list_attachments": (
            tool_list_attachments,
            lambda a, c: (a.get("type"), a.get("query"), a.get("limit", 20)),
        ),
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
    prompt = "Describe this image in detail. Include all visible text, numbers, labels, and notable visual elements."
    if caption:
        prompt = f"The user sent this image with caption: '{caption}'. Describe what you see in detail, including all visible text, numbers, and labels."

    try:
        print(f"[preprocess_image] Processing {file_path}", flush=True)
        image_bytes = Path(file_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        mime_type = get_mime_type(Path(file_path))

        response = await vision_complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                    ],
                }
            ],
            timeout=60,
        )
        result = response.choices[0].message.content.strip()
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

        response = await vision_complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this audio exactly and completely. Output only the transcript, nothing else.",
                        },
                        {
                            "type": "file",
                            "file": {"file_data": f"data:{mime_type};base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        result = response.choices[0].message.content.strip()
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

        response = await vision_complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Briefly summarize this PDF document. Include: page count, document type/purpose, key dates, amounts, names, and main points. Keep it concise (2-4 sentences).",
                        },
                        {
                            "type": "file",
                            "file": {"file_data": f"data:application/pdf;base64,{b64}"},
                        },
                    ],
                }
            ],
            timeout=120,
        )
        result = response.choices[0].message.content.strip()
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
    user_id: int | None = None,
    interrupt: asyncio.Event | None = None,
) -> str | None:
    """Run the Opus agent loop with tools. Returns None if interrupted."""
    system_prompt = extract_system_prompt()
    history = get_session(chat_id)

    # Inject time and user identity into the current user message
    # This preserves system prompt caching while giving the model (and history) context
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
    user_name = USER_NAMES.get(user_id, "Unknown") if user_id else "Unknown"
    user_message = f"[{current_time}] [{user_name}] {user_message}"

    # Build messages for API
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Hash the history prefix for consistency verification
    history_hash = hash_messages(history) if history else "empty"
    print(
        f"[run_agent] chat_id={chat_id}, history={len(history)}, messages={len(messages)}, history_hash={history_hash}",
        flush=True,
    )

    def save_partial():
        """Save partial work to session when interrupted."""
        nonlocal history
        history_len = len(history)
        add_message_to_session(chat_id, {"role": "user", "content": user_message})
        for m in messages[history_len + 2 :]:
            add_message_to_session(chat_id, m)

    response = None

    for iteration in range(MAX_TOOL_ITERATIONS):
        # Check for interrupt before API call
        if interrupt and interrupt.is_set():
            save_partial()
            return None

        # Log cumulative prefix hashes for each message (for cache consistency verification)
        prefix_hashes = [hash_messages(messages[: i + 1]) for i in range(len(messages))]
        print(
            f"[run_agent] API call iteration={iteration}, hashes={','.join(prefix_hashes)}",
            flush=True,
        )

        try:
            response = await reasoning_complete(
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"[run_agent] API error: {e}", flush=True)
            # Return error to user
            return f"API error: {e}"

        msg = response.choices[0].message

        # Send intermediate text to user immediately (if present alongside tool calls)
        if msg.content and msg.tool_calls:
            chunks = (
                [msg.content[i : i + 4096] for i in range(0, len(msg.content), 4096)]
                if len(msg.content) > 4096
                else [msg.content]
            )
            for chunk in chunks:
                try:
                    await _bot.send_message(
                        chat_id=chat_id, text=chunk, parse_mode="HTML"
                    )
                except TelegramError:
                    await _bot.send_message(chat_id=chat_id, text=chunk)

        # Check if done (no tool calls)
        if not msg.tool_calls:
            # Check for interrupt before returning final result
            if interrupt and interrupt.is_set():
                save_partial()
                return None

            # Save to session: user + all intermediate messages + final response (actual content)
            history_len = len(history)
            add_message_to_session(chat_id, {"role": "user", "content": user_message})
            # Skip: system (1) + history (history_len) + initial user msg (1) = history_len + 2
            for m in messages[history_len + 2 :]:
                add_message_to_session(chat_id, m)
            add_message_to_session(chat_id, build_assistant_msg(msg))  # Actual response

            session_now = get_session(chat_id)
            print(
                f"[run_agent] Saved {len(session_now) - history_len} new messages (total: {len(session_now)}, hash: {hash_messages(session_now)})",
                flush=True,
            )

            # Return final response only (ignore intermediate "thinking" text)
            # Return None if no content (e.g., tool-only response) - caller will skip sending
            return msg.content or None

        # Process tool calls - build message preserving all fields
        assistant_msg = build_assistant_msg(msg)
        messages.append(assistant_msg)

        # Execute ALL tool calls in parallel before checking interrupt
        # (tools may have side effects, API requires all tool_calls to have matching results)
        async def run_tool(tc):
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments or "{}")
            result = await execute_tool(fn_name, fn_args, chat_id)
            return {"role": "tool", "tool_call_id": tc.id, "content": result}

        tool_results = await asyncio.gather(*[run_tool(tc) for tc in msg.tool_calls])
        messages.extend(tool_results)

        # Check for interrupt after all tools complete
        if interrupt and interrupt.is_set():
            save_partial()
            return None

    # Max iterations reached - save actual state (no synthetic message)
    history_len = len(history)
    add_message_to_session(chat_id, {"role": "user", "content": user_message})
    for m in messages[history_len + 2 :]:
        add_message_to_session(chat_id, m)

    session_now = get_session(chat_id)
    print(
        f"[run_agent] Max iterations reached. Saved {len(session_now) - history_len} messages (total: {len(session_now)}, hash: {hash_messages(session_now)})",
        flush=True,
    )

    # Display-only warning (not saved to session)
    last_content = response.choices[0].message.content if response else ""
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

        # Save metadata
        save_attachment_metadata(
            attachment_id,
            attachment_type,
            description,
            chat_id,
            user_id,
            original_filename,
        )

        # Format message based on type
        if attachment_type == "voice":
            return f'[Voice {attachment_id}: "{description}"]'
        elif attachment_type == "audio":
            fname = f" {original_filename}" if original_filename else ""
            return f'[Audio {attachment_id}{fname}: "{description}"]'
        elif attachment_type == "image":
            fname = f" {original_filename}" if original_filename else ""
            return f"[Image {attachment_id}{fname}: {description}]"
        elif attachment_type == "pdf":
            fname = f" {original_filename}" if original_filename else ""
            return f"[PDF {attachment_id}{fname}: {description}]"
        else:
            return f"[Attachment {attachment_id}: {description}]"

    # Handle voice messages
    if msg.voice:
        result = await process_attachment(
            msg.voice.file_id, ".ogg", "voice", preprocess_audio
        )
        if result:
            text = f"{result}\n\n{text}".strip()
        else:
            text = f"[Voice message received but couldn't transcribe]\n\n{text}".strip()

    # Handle photos (sent as photos, not documents)
    if msg.photo:
        photo = msg.photo[-1]  # Highest resolution
        result = await process_attachment(
            photo.file_id, ".jpg", "image", preprocess_image
        )
        if result:
            text = f"{result}\n\n{text}".strip()
        else:
            text = f"[Image received but couldn't process]\n\n{text}".strip()

    # Handle documents (images, PDFs, audio files)
    if msg.document:
        doc = msg.document
        mime = doc.mime_type or ""
        original_filename = doc.file_name
        ext = Path(original_filename).suffix.lower() if original_filename else ""

        if mime.startswith("image/"):
            # Image sent as document
            ext = ext or ".jpg"
            result = await process_attachment(
                doc.file_id, ext, "image", preprocess_image, original_filename
            )
            if result:
                text = f"{result}\n\n{text}".strip()
            else:
                text = f"[Image received but couldn't process]\n\n{text}".strip()

        elif mime == "application/pdf" or ext == ".pdf":
            # PDF document
            result = await process_attachment(
                doc.file_id, ".pdf", "pdf", preprocess_pdf, original_filename
            )
            if result:
                text = f"{result}\n\n{text}".strip()
            else:
                text = f"[PDF received but couldn't process: {original_filename}]\n\n{text}".strip()

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
            result = await process_attachment(
                doc.file_id, ext, "audio", preprocess_audio, original_filename
            )
            if result:
                text = f"{result}\n\n{text}".strip()
            else:
                text = f"[Audio received but couldn't transcribe: {original_filename}]\n\n{text}".strip()

        else:
            # Unsupported document type
            text = (
                f"[Document received: {original_filename} ({mime})]\n\n{text}".strip()
            )

    if not text:
        return

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    get_session(chat_id)  # Ensure session exists
    session = sessions[chat_id]

    # Signal interrupt if lock is already held (another run in progress)
    if session["lock"].locked():
        session["interrupt"].set()

    async with session["lock"]:
        session["interrupt"].clear()

        response = await run_agent(
            text,
            chat_id,
            user_id,
            interrupt=session["interrupt"],
        )

        # Interrupted or no content - nothing to send
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
        # Run agent with the wakeup prompt
        # Use a synthetic user_id (0) to indicate this is a scheduled message
        response = await run_agent(
            f"[SCHEDULED WAKEUP] {prompt}",
            chat_id,
            0,  # System/scheduled user ID
        )

        # Send the response
        chunks = (
            [response[i : i + 4096] for i in range(0, len(response), 4096)]
            if len(response) > 4096
            else [response]
        )

        for chunk in chunks:
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")
            except TelegramError as e:
                print(f"[scheduler] HTML parse failed: {e}", flush=True)
                await bot.send_message(chat_id=chat_id, text=chunk)

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

            # Daily rotation check at ROTATION_HOUR (4am)
            global _last_rotation_check
            today = now.strftime("%Y-%m-%d")

            if now.hour == ROTATION_HOUR and _last_rotation_check != today:
                _last_rotation_check = today
                stream_file = DATA_DIR / "stream.txt"

                if (
                    stream_file.exists()
                    and stream_file.stat().st_size > ROTATION_THRESHOLD
                ):
                    print(
                        "[scheduler] Stream exceeds threshold, starting rotation...",
                        flush=True,
                    )
                    try:
                        from rotation import rotate_stream

                        result = await rotate_stream(force=True)

                        # Notify owner
                        if OWNER_ID:
                            try:
                                await bot.send_message(
                                    chat_id=OWNER_ID, text=f"🔄 {result}"
                                )
                            except Exception as e:
                                print(
                                    f"[scheduler] Failed to notify owner: {e}",
                                    flush=True,
                                )
                    except Exception as e:
                        print(f"[scheduler] Rotation failed: {e}", flush=True)

            # Check for expired sessions
            now_ts = time.time()
            expired_chats = []
            for chat_id, session in list(sessions.items()):
                if now_ts - session["last_used"] > SESSION_TIMEOUT:
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
                sessions.pop(chat_id, None)

        except Exception as e:
            print(f"[scheduler] Error in scheduler loop: {e}", flush=True)

        await asyncio.sleep(SCHEDULER_INTERVAL)


async def post_init(app) -> None:
    """Called after the application is initialized."""
    global _bot
    _bot = app.bot
    # Start the scheduler as a background task
    asyncio.create_task(scheduler_loop(app))


async def post_shutdown(app) -> None:
    """Called after the application is shut down - cleanup resources."""
    # Close litellm's async HTTP clients to avoid the warning:
    # "RuntimeWarning: coroutine 'close_litellm_async_clients' was never awaited"
    try:
        await litellm.aclient_session_cleanup()
    except Exception as e:
        print(f"[shutdown] Error cleaning up litellm clients: {e}", flush=True)


def main() -> None:
    # Ensure attachments directory exists
    ATTACHMENTS_DIR.mkdir(exist_ok=True)

    app = (
        Application.builder()
        .token(TOKEN)
        .concurrent_updates(True)  # Allow handlers to run concurrently
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_session))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))

    print(f"Bot starting... allowed users: {ALLOWED_USERS or 'all'}")
    print(f"Model: {REASONING['model']}")
    print(f"Session timeout: {SESSION_TIMEOUT // 3600}h")
    print(f"Scheduler interval: {SCHEDULER_INTERVAL}s")
    print(f"Attachments dir: {ATTACHMENTS_DIR}")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
