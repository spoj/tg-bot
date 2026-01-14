"""
E2B Sandbox Manager - Sandbox lifecycle management for code execution.

Each chat session gets a fresh E2B sandbox for running arbitrary code.
Sandboxes are created lazily on first use and auto-expire after 5 mins of inactivity.
Timeout is renewed on each e2b operation via set_timeout().
"""

from dataclasses import dataclass

from e2b import AsyncSandbox

# Sandbox timeout: 5 minutes (renewed on each operation)
SANDBOX_TIMEOUT_SECONDS = 300


@dataclass
class SandboxSession:
    """Represents a chat's sandbox session."""

    chat_id: int
    sandbox: AsyncSandbox


class SandboxManager:
    """Manages E2B sandboxes for multiple chats."""

    def __init__(self):
        self.sessions: dict[int, SandboxSession] = {}

    async def _renew_timeout(self, session: SandboxSession):
        """Renew E2B timeout on sandbox."""
        try:
            await session.sandbox.set_timeout(SANDBOX_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"[e2b] Failed to renew timeout: {e}", flush=True)

    async def get_or_create_session(self, chat_id: int) -> tuple[SandboxSession, bool]:
        """Get existing session or create a new one. Returns (session, is_new)."""
        if chat_id in self.sessions:
            session = self.sessions[chat_id]
            # Check if sandbox is still running
            try:
                if await session.sandbox.is_running():
                    await self._renew_timeout(session)
                    return session, False
            except Exception:
                pass
            # Sandbox died, clean up
            self.sessions.pop(chat_id, None)

        print(f"[e2b] Creating new sandbox for chat {chat_id}", flush=True)
        sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT_SECONDS)

        # Create working directory
        await sandbox.commands.run("mkdir -p /home/user/workspace", timeout=5)

        session = SandboxSession(chat_id=chat_id, sandbox=sandbox)
        self.sessions[chat_id] = session
        return session, True

    async def run_command(self, chat_id: int, command: str, timeout: int = 120) -> dict:
        """Run a shell command in the chat's sandbox. Returns dict with is_new_sandbox flag."""
        session, is_new = await self.get_or_create_session(chat_id)

        try:
            result = await session.sandbox.commands.run(
                command,
                timeout=timeout,
                cwd="/home/user/workspace",
            )
            await self._renew_timeout(session)
            return {
                "success": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "is_new_sandbox": is_new,
            }
        except Exception as e:
            print(f"[e2b] Command error for chat {chat_id}: {e}", flush=True)
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "is_new_sandbox": is_new,
            }

    async def upload_file(
        self, chat_id: int, filename: str, content: bytes
    ) -> tuple[str, bool]:
        """Upload a file to the sandbox workspace. Returns (remote_path, is_new_sandbox)."""
        session, is_new = await self.get_or_create_session(chat_id)
        path = f"/home/user/workspace/{filename}"
        await session.sandbox.files.write(path, content)
        await self._renew_timeout(session)
        print(
            f"[e2b] Uploaded {filename} ({len(content)} bytes) for chat {chat_id}",
            flush=True,
        )
        return path, is_new

    async def read_file(self, chat_id: int, path: str) -> tuple[str | None, str]:
        """Read a text file from the sandbox. Returns (content, error)."""
        if chat_id not in self.sessions:
            return (
                None,
                "No active sandbox (expires after 5 mins idle). Use e2b_upload or e2b_run to start a new session.",
            )

        session = self.sessions[chat_id]
        if not path.startswith("/"):
            path = f"/home/user/workspace/{path}"

        try:
            content = await session.sandbox.files.read(path)
            await self._renew_timeout(session)
            # E2B returns bytes or str depending on version/context
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace"), ""
            return str(content), ""
        except Exception as e:
            return None, str(e)

    async def download_file(self, chat_id: int, path: str) -> tuple[bytes | None, str]:
        """Download a binary file from the sandbox. Returns (content, error)."""
        if chat_id not in self.sessions:
            return (
                None,
                "No active sandbox (expires after 5 mins idle). Use e2b_upload or e2b_run to start a new session.",
            )

        session = self.sessions[chat_id]
        if not path.startswith("/"):
            path = f"/home/user/workspace/{path}"

        try:
            content = await session.sandbox.files.read(path, format="bytes")
            await self._renew_timeout(session)
            return bytes(content), ""
        except Exception as e:
            return None, str(e)


# Global singleton instance
sandbox_manager = SandboxManager()
