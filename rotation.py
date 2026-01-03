"""Stream rotation and dream time analysis."""

import asyncio
import fcntl
import os
import random
from datetime import datetime
from pathlib import Path

from litellm import acompletion

# Config (same env vars as bot.py)
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / "life"))
AGENT_MODEL = os.environ.get("AGENT_MODEL", "openrouter/anthropic/claude-opus-4.5")
ROTATION_THRESHOLD = 100 * 1024  # 100kb
ACCESS_SAMPLE_SIZE = 5


def sample_access_log(n: int = ACCESS_SAMPLE_SIZE) -> list[str]:
    """Random sample of queries from ACCESS.log."""
    access_log = DATA_DIR / "ACCESS.log"
    if not access_log.exists():
        return []

    content = access_log.read_text().strip()
    if not content:
        return []

    lines = content.split("\n")

    # Parse queries from log format: "2026-01-03T20:33:00 tg "query""
    queries = []
    for line in lines:
        if '"' in line:
            try:
                query = line.split('"')[1]
                queries.append(query)
            except IndexError:
                pass

    if len(queries) <= n:
        return queries
    return random.sample(queries, n)


async def extract_commitments(content: str) -> str:
    """Step 1: Extract open TODOs and calendar from content."""
    response = await acompletion(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Extract all open commitments from this stream file:
- Open TODOs (lines starting with "TODO:" that are not marked with checkmark)
- Calendar items (+cal entries)

Output in the same format as the original (plain text, one per line).
If none found, output "None."

Stream content:
{content}""",
            }
        ],
        timeout=600,
    )
    return response.choices[0].message.content.strip()


async def pairwise_analysis(
    fresh: tuple[str, str],  # (date, content) - FIRST for caching
    older: tuple[str, str],
    access_sample: list[str],
) -> str:
    """Step 2: Compare fresh file with older archive."""
    fresh_date, fresh_content = fresh
    older_date, older_content = older

    access_context = ""
    if access_sample:
        access_context = f"""
Recently queried topics (sample):
{chr(10).join(f"- {q}" for q in access_sample)}
"""

    response = await acompletion(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Compare these two stream files and find connections, contradictions, or notable observations.

Note: There may be time gaps between these files that you're not seeing. That's okay - focus on what you can observe from these two.
{access_context}
--- Stream from {fresh_date} (most recent) ---
{fresh_content}

--- Stream from {older_date} ---
{older_content}

Observations:""",
            }
        ],
        timeout=600,
    )
    return response.choices[0].message.content.strip()


async def synthesize(observations: list[str]) -> str:
    """Step 3: Combine pairwise observations into coherent findings."""
    numbered = "\n\n".join(f"[{i + 1}]\n{obs}" for i, obs in enumerate(observations))

    response = await acompletion(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Synthesize these pairwise observations into coherent findings.

Look for:
- Patterns across time
- Contradictions or stale information
- Loose threads worth noting
- Evolution of ideas or situations

Keep it concise and actionable.

Observations from pairwise comparisons:
{numbered}

Synthesized findings:""",
            }
        ],
        timeout=600,
    )
    return response.choices[0].message.content.strip()


async def rotate_stream(force: bool = False) -> str:
    """Main entry. Archive, run dream time, return summary.

    Args:
        force: If True, rotate regardless of size threshold.
    """
    stream_file = DATA_DIR / "stream.txt"
    lock_file = DATA_DIR / ".stream.txt.lock"

    # --- Pre-check: Refuse if rotated recently (idempotency) ---
    today = datetime.now().strftime("%Y-%m-%d")
    today_archive = DATA_DIR / f"stream-{today}.txt"
    if today_archive.exists():
        return f"Rotation skipped: {today_archive.name} already exists."

    archives = sorted(DATA_DIR.glob("stream-*.txt"))
    if archives:
        latest_archive = archives[-1]
        latest_mtime = latest_archive.stat().st_mtime
        hours_since_last = (datetime.now().timestamp() - latest_mtime) / 3600
        if hours_since_last < 24:
            return f"Rotation skipped: last archive {latest_archive.name} is only {hours_since_last:.1f}h old (< 24h)."

    # --- Phase 1: Read with lock ---
    with open(lock_file, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if not stream_file.exists():
                return "No stream.txt to rotate."

            size = stream_file.stat().st_size
            if not force and size < ROTATION_THRESHOLD:
                return f"Stream size {size // 1024}kb < 100kb. No rotation needed."

            original_mtime = stream_file.stat().st_mtime
            fresh_content = stream_file.read_text()

            # Archive immediately while holding lock
            archive_path = today_archive
            archive_path.write_text(fresh_content)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    # Lock released

    fresh_date = archive_path.stem.replace("stream-", "")
    print(f"[rotation] Archived to {archive_path.name}", flush=True)

    # --- Phase 2: LLM calls (no lock, may take minutes) ---

    # Step 1: Extract commitments
    print("[rotation] Extracting commitments...", flush=True)
    try:
        commitments = await extract_commitments(fresh_content)
    except Exception as e:
        print(f"[rotation] extract_commitments failed: {e}", flush=True)
        commitments = "(extraction failed)"

    # Find older archives (excluding the one we just created)
    archives = sorted(DATA_DIR.glob("stream-*.txt"))
    older_archives = [a for a in archives if a != archive_path]

    # Step 2: Pairwise analysis
    observations = []
    if older_archives:
        total = len(older_archives)
        print(
            f"[rotation] Running pairwise analysis with {total} archives...", flush=True
        )
        access_queries = sample_access_log()

        async def run_pairwise(idx: int, older: Path) -> str | Exception:
            """Run single pairwise analysis with logging."""
            older_date = older.stem.replace("stream-", "")
            print(
                f"[rotation] Pairwise {idx + 1}/{total}: {older.name} starting...",
                flush=True,
            )
            try:
                older_content = older.read_text()
                result = await pairwise_analysis(
                    (fresh_date, fresh_content),  # Fresh first for caching
                    (older_date, older_content),
                    access_queries,
                )
                print(
                    f"[rotation] Pairwise {idx + 1}/{total}: {older.name} done",
                    flush=True,
                )
                return result
            except Exception as e:
                print(
                    f"[rotation] Pairwise {idx + 1}/{total}: {older.name} failed: {e}",
                    flush=True,
                )
                return e

        results = await asyncio.gather(
            *[run_pairwise(i, a) for i, a in enumerate(older_archives)]
        )
        for r in results:
            if not isinstance(r, Exception):
                observations.append(r)

    # Step 3: Synthesize
    findings = ""
    if observations:
        print("[rotation] Synthesizing findings...", flush=True)
        try:
            findings = await synthesize(observations)
        except Exception as e:
            print(f"[rotation] synthesis failed: {e}", flush=True)
            findings = "(synthesis failed)"
    else:
        findings = "(no older archives to compare)"

    # --- Phase 3: Write with lock, check for concurrent modification ---
    new_content = f"""## {fresh_date}

## Open Commitments
{commitments}

## Dream Time
{findings}
"""

    with open(lock_file, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current_mtime = stream_file.stat().st_mtime if stream_file.exists() else 0

            if current_mtime != original_mtime:
                # Stream was modified during our LLM calls - append instead of overwrite
                print(
                    "[rotation] Stream modified during rotation, appending...",
                    flush=True,
                )
                current_content = stream_file.read_text()
                stream_file.write_text(
                    new_content
                    + "\n---\n(content added during rotation)\n"
                    + current_content
                )
            else:
                # No modification - safe to overwrite
                stream_file.write_text(new_content)

            # Clear ACCESS.log
            access_log = DATA_DIR / "ACCESS.log"
            if access_log.exists():
                access_log.write_text("")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    summary = (
        f"Rotated to {archive_path.name}. {len(older_archives)} archives compared."
    )
    print(f"[rotation] {summary}", flush=True)
    return summary
