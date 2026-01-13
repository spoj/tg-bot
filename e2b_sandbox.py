"""
E2B Sandbox Manager - Sandbox lifecycle management for code execution.

Each chat session gets a fresh E2B sandbox (1hr TTL) for running arbitrary code.
Sandboxes are created lazily on first use and destroyed on timeout/restart.
"""

import asyncio
import time
from dataclasses import dataclass, field

from e2b import AsyncSandbox

# Sandbox timeout: 1 hour (matches E2B TTL)
SANDBOX_TIMEOUT_SECONDS = 3600


@dataclass
class SandboxSession:
    """Represents a chat's sandbox session."""

    chat_id: int
    sandbox: AsyncSandbox
    created_at: float = field(default_factory=time.time)

    @property
    def remaining_seconds(self) -> float:
        return max(0, SANDBOX_TIMEOUT_SECONDS - (time.time() - self.created_at))

    @property
    def is_expired(self) -> bool:
        return self.remaining_seconds <= 0


class SandboxManager:
    """Manages E2B sandboxes for multiple chats."""

    def __init__(self):
        self.sessions: dict[int, SandboxSession] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def start(self):
        """Start the background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        print("[e2b] SandboxManager started", flush=True)

    async def stop(self):
        """Stop manager and cleanup all sessions."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        for session in list(self.sessions.values()):
            await self._destroy_session(session)

        print("[e2b] SandboxManager stopped", flush=True)

    async def _cleanup_loop(self):
        """Periodically clean up expired sessions."""
        while True:
            await asyncio.sleep(60)
            expired = [s for s in self.sessions.values() if s.is_expired]
            for session in expired:
                print(f"[e2b] Session expired for chat {session.chat_id}", flush=True)
                await self._destroy_session(session)

    async def _destroy_session(self, session: SandboxSession):
        """Destroy a session and its sandbox."""
        try:
            await session.sandbox.kill()
        except Exception as e:
            print(f"[e2b] Error killing sandbox: {e}", flush=True)
        self.sessions.pop(session.chat_id, None)

    async def get_or_create_session(self, chat_id: int) -> SandboxSession:
        """Get existing session or create a new one."""
        if chat_id in self.sessions:
            session = self.sessions[chat_id]
            if not session.is_expired:
                return session
            await self._destroy_session(session)

        print(f"[e2b] Creating new sandbox for chat {chat_id}", flush=True)
        sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT_SECONDS)

        # Create working directory and install common packages
        await sandbox.commands.run("mkdir -p /home/user/workspace", timeout=5)

        session = SandboxSession(chat_id=chat_id, sandbox=sandbox)
        self.sessions[chat_id] = session
        return session

    async def run_command(self, chat_id: int, command: str, timeout: int = 120) -> dict:
        """Run a shell command in the chat's sandbox."""
        session = await self.get_or_create_session(chat_id)

        try:
            result = await session.sandbox.commands.run(
                command,
                timeout=timeout,
                cwd="/home/user/workspace",
            )
            return {
                "success": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            }
        except Exception as e:
            print(f"[e2b] Command error for chat {chat_id}: {e}", flush=True)
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
            }

    async def upload_file(self, chat_id: int, filename: str, content: bytes) -> str:
        """Upload a file to the sandbox workspace. Returns the remote path."""
        session = await self.get_or_create_session(chat_id)
        path = f"/home/user/workspace/{filename}"
        await session.sandbox.files.write(path, content)
        print(
            f"[e2b] Uploaded {filename} ({len(content)} bytes) for chat {chat_id}",
            flush=True,
        )
        return path

    async def read_file(self, chat_id: int, path: str) -> tuple[str | None, str]:
        """Read a text file from the sandbox. Returns (content, error)."""
        if chat_id not in self.sessions:
            return None, "No active sandbox session. Upload a file first."

        session = self.sessions[chat_id]
        if not path.startswith("/"):
            path = f"/home/user/workspace/{path}"

        try:
            content = await session.sandbox.files.read(path)
            return content.decode("utf-8", errors="replace"), ""
        except Exception as e:
            return None, str(e)

    async def download_file(self, chat_id: int, path: str) -> tuple[bytes | None, str]:
        """Download a binary file from the sandbox. Returns (content, error)."""
        if chat_id not in self.sessions:
            return None, "No active sandbox session."

        session = self.sessions[chat_id]
        if not path.startswith("/"):
            path = f"/home/user/workspace/{path}"

        try:
            content = await session.sandbox.files.read(path, format="bytes")
            return bytes(content), ""
        except Exception as e:
            return None, str(e)

    async def destroy_session(self, chat_id: int):
        """Manually destroy a chat's sandbox session."""
        if chat_id in self.sessions:
            await self._destroy_session(self.sessions[chat_id])
            print(f"[e2b] Session destroyed for chat {chat_id}", flush=True)

    def has_session(self, chat_id: int) -> bool:
        """Check if a chat has an active sandbox session."""
        return chat_id in self.sessions and not self.sessions[chat_id].is_expired


# Global singleton instance
sandbox_manager = SandboxManager()
