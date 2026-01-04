"""Tests for bot.py - focusing on pure/isolated functions."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from bot import (
    parse_wake_time,
    tool_random_pick,
    tool_stream_replace,
    load_wakeups,
    save_wakeups,
    execute_tool,
    get_system_prompt,
    DATA_DIR,
)


class TestParseWakeTime:
    """Tests for parse_wake_time() - time parsing from various formats."""

    def test_iso_format_datetime(self):
        result = parse_wake_time("2026-01-15T09:00")
        assert result == datetime(2026, 1, 15, 9, 0)

    def test_iso_format_with_seconds(self):
        result = parse_wake_time("2026-01-15T09:30:45")
        assert result == datetime(2026, 1, 15, 9, 30, 45)

    def test_iso_format_space_separator(self):
        result = parse_wake_time("2026-01-15 14:30")
        assert result == datetime(2026, 1, 15, 14, 30)

    def test_relative_minutes(self):
        before = datetime.now()
        result = parse_wake_time("in 30 minutes")
        after = datetime.now()

        expected_min = before + timedelta(minutes=30)
        expected_max = after + timedelta(minutes=30)
        assert expected_min <= result <= expected_max

    def test_relative_hours(self):
        before = datetime.now()
        result = parse_wake_time("in 2 hours")
        after = datetime.now()

        expected_min = before + timedelta(hours=2)
        expected_max = after + timedelta(hours=2)
        assert expected_min <= result <= expected_max

    def test_relative_days(self):
        before = datetime.now()
        result = parse_wake_time("in 3 days")
        after = datetime.now()

        expected_min = before + timedelta(days=3)
        expected_max = after + timedelta(days=3)
        assert expected_min <= result <= expected_max

    def test_relative_mins_abbreviation(self):
        result = parse_wake_time("in 5 mins")
        assert result is not None
        assert result > datetime.now()

    def test_relative_hr_abbreviation(self):
        result = parse_wake_time("in 1 hr")
        assert result is not None

    def test_tomorrow_with_time(self):
        result = parse_wake_time("tomorrow 9am")
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.date() == tomorrow.date()
        assert result.hour == 9
        assert result.minute == 0

    def test_tomorrow_pm(self):
        result = parse_wake_time("tomorrow 2pm")
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.date() == tomorrow.date()
        assert result.hour == 14

    def test_tomorrow_24h_format(self):
        result = parse_wake_time("tomorrow 14:30")
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.date() == tomorrow.date()
        assert result.hour == 14
        assert result.minute == 30

    def test_tomorrow_no_time_defaults_9am(self):
        result = parse_wake_time("tomorrow")
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.date() == tomorrow.date()
        assert result.hour == 9

    def test_bare_time_future_today(self):
        # If we ask for a time later today, should be today
        now = datetime.now()
        future_hour = (now.hour + 2) % 24
        if future_hour > now.hour:  # Only test if it's still today
            result = parse_wake_time(f"{future_hour}:00")
            assert result.date() == now.date()
            assert result.hour == future_hour

    def test_bare_time_past_schedules_tomorrow(self):
        # If we ask for a time that's passed, should be tomorrow
        now = datetime.now()
        past_hour = (now.hour - 2) % 24
        result = parse_wake_time(f"{past_hour}:00")
        # Should be tomorrow if the time has passed
        if past_hour < now.hour:
            tomorrow = now + timedelta(days=1)
            assert result.date() == tomorrow.date()

    def test_invalid_format_returns_none(self):
        assert parse_wake_time("garbage") is None
        assert parse_wake_time("next tuesday") is None  # Not implemented
        assert parse_wake_time("") is None

    def test_case_insensitive(self):
        result1 = parse_wake_time("Tomorrow 9AM")
        result2 = parse_wake_time("tomorrow 9am")
        assert result1 == result2


class TestToolRandomPick:
    """Tests for tool_random_pick()."""

    def test_pick_one_from_list(self):
        items = ["a", "b", "c"]
        result = tool_random_pick(items, 1)
        assert result in items

    def test_pick_multiple(self):
        items = ["a", "b", "c", "d", "e"]
        result = tool_random_pick(items, 3)
        lines = result.split("\n")
        assert len(lines) == 3
        # Check format "1. item"
        assert lines[0].startswith("1. ")

    def test_pick_default_is_one(self):
        items = ["a", "b", "c"]
        result = tool_random_pick(items)
        assert result in items
        assert "\n" not in result  # Single item, no newlines

    def test_empty_list_error(self):
        result = tool_random_pick([])
        assert "Error" in result

    def test_n_less_than_one_error(self):
        result = tool_random_pick(["a", "b"], 0)
        assert "Error" in result

    def test_n_greater_than_list_caps(self):
        items = ["a", "b", "c"]
        result = tool_random_pick(items, 10)
        lines = result.split("\n")
        assert len(lines) == 3  # Capped at list size

    def test_no_duplicates(self):
        items = ["a", "b", "c", "d", "e"]
        result = tool_random_pick(items, 5)
        lines = result.split("\n")
        picked = [line.split(". ", 1)[1] for line in lines]
        assert len(picked) == len(set(picked))  # All unique


class TestWakeupsPersistence:
    """Tests for load_wakeups() and save_wakeups()."""

    def test_load_nonexistent_file_returns_empty(self):
        with patch("bot.WAKEUPS_FILE", Path("/nonexistent/wakeups.json")):
            result = load_wakeups()
            assert result == []

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with patch("bot.WAKEUPS_FILE", temp_path):
                wakeups = [
                    {
                        "id": "abc123",
                        "wake_time": "2026-01-15T09:00",
                        "prompt": "test",
                        "chat_id": 123,
                    },
                    {
                        "id": "def456",
                        "wake_time": "2026-01-16T10:00",
                        "prompt": "test2",
                        "chat_id": 456,
                    },
                ]
                save_wakeups(wakeups)
                loaded = load_wakeups()
                assert loaded == wakeups
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_corrupted_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            temp_path = Path(f.name)

        try:
            with patch("bot.WAKEUPS_FILE", temp_path):
                result = load_wakeups()
                assert result == []
        finally:
            temp_path.unlink(missing_ok=True)


class TestExecuteToolDispatch:
    """Tests that execute_tool correctly dispatches to handlers."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        result = await execute_tool("nonexistent_tool", {}, chat_id=123)
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_read_stream_tail_default_n(self):
        with patch("bot.tool_read_stream_tail") as mock:
            mock.return_value = "stream content"
            await execute_tool("read_stream_tail", {}, chat_id=123)
            mock.assert_called_once_with(50)

    @pytest.mark.asyncio
    async def test_read_stream_tail_custom_n(self):
        with patch("bot.tool_read_stream_tail") as mock:
            mock.return_value = "stream content"
            await execute_tool("read_stream_tail", {"n": 100}, chat_id=123)
            mock.assert_called_once_with(100)

    @pytest.mark.asyncio
    async def test_schedule_wakeup_passes_chat_id(self):
        with patch("bot.tool_schedule_wakeup") as mock:
            mock.return_value = "scheduled"
            await execute_tool(
                "schedule_wakeup",
                {"wake_time": "in 1 hour", "prompt": "test"},
                chat_id=999,
            )
            mock.assert_called_once()
            # Check chat_id was passed (4th argument)
            assert mock.call_args[0][3] == 999

    @pytest.mark.asyncio
    async def test_list_wakeups_passes_chat_id(self):
        with patch("bot.tool_list_wakeups") as mock:
            mock.return_value = "no wakeups"
            await execute_tool("list_wakeups", {}, chat_id=888)
            mock.assert_called_once_with(888)

    @pytest.mark.asyncio
    async def test_random_pick_dispatch(self):
        with patch("bot.tool_random_pick") as mock:
            mock.return_value = "picked"
            await execute_tool(
                "random_pick", {"items": ["a", "b"], "n": 1}, chat_id=123
            )
            mock.assert_called_once_with(["a", "b"], 1)


class TestGetSystemPrompt:
    """Tests for get_system_prompt()."""

    def test_contains_core_instructions(self):
        """System prompt contains the essential life agent instructions."""
        result = get_system_prompt()
        assert "Life Agent Instructions" in result
        assert "memory" in result.lower()
        assert "stream.txt" in result

    def test_contains_telegram_formatting(self):
        """System prompt includes Telegram HTML formatting instructions."""
        result = get_system_prompt()
        assert "Telegram" in result
        assert "<b>" in result  # HTML formatting instructions

    def test_contains_owner_placeholder(self):
        """System prompt has owner name substituted."""
        result = get_system_prompt()
        # Should not contain the literal placeholder
        assert "{owner_name}" not in result


class TestToolStreamReplace:
    """Tests for tool_stream_replace()."""

    def test_replace_in_last_100_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stream_file = tmpdir / "stream.txt"
            stream_file.write_text("# stream\n\nLine 1\nLine 2\nLine 3\n")

            with patch("bot.DATA_DIR", tmpdir):
                result = tool_stream_replace("Line 2", "Updated Line 2")
                assert "Replaced" in result
                content = stream_file.read_text()
                assert "Updated Line 2" in content
                assert "Line 2" not in content or "Updated Line 2" in content

    def test_not_found_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stream_file = tmpdir / "stream.txt"
            stream_file.write_text("# stream\n\nLine 1\nLine 2\n")

            with patch("bot.DATA_DIR", tmpdir):
                result = tool_stream_replace("Nonexistent line", "New")
                assert "Error" in result
                assert "not found" in result

    def test_multiple_matches_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stream_file = tmpdir / "stream.txt"
            stream_file.write_text("# stream\n\nDuplicate\nDuplicate\n")

            with patch("bot.DATA_DIR", tmpdir):
                result = tool_stream_replace("Duplicate", "New")
                assert "Error" in result
                assert "2 times" in result

    def test_outside_100_line_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stream_file = tmpdir / "stream.txt"
            # Create file with 150 lines, target text in first 50
            lines = ["# stream"] + [f"Line {i}" for i in range(1, 150)]
            lines[10] = "Old text to replace"  # Line 10 - outside last 100
            stream_file.write_text("\n".join(lines))

            with patch("bot.DATA_DIR", tmpdir):
                result = tool_stream_replace("Old text to replace", "New text")
                assert "Error" in result
                assert "not found in last 100" in result

    def test_within_100_line_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stream_file = tmpdir / "stream.txt"
            # Create file with 150 lines, target text in last 50
            lines = ["# stream"] + [f"Line {i}" for i in range(1, 150)]
            lines[140] = "Text to replace"  # Line 140 - inside last 100
            stream_file.write_text("\n".join(lines))

            with patch("bot.DATA_DIR", tmpdir):
                result = tool_stream_replace("Text to replace", "Replaced text")
                assert "Replaced" in result
                content = stream_file.read_text()
                assert "Replaced text" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
