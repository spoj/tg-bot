"""Tests for agent_loop.py - generic agent loop with hooks."""

import asyncio
import pytest

from agent_loop import (
    run_agent_loop,
    LoopHooks,
    AgentResult,
    accumulate_usage,
    process_tool_calls,
)


class TestAccumulateUsage:
    """Tests for usage accumulation."""

    def test_basic_accumulation(self):
        total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read": 10,
            "cache_write": 5,
        }
        accumulate_usage(total, usage)
        assert total == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read": 10,
            "cache_write": 5,
        }

    def test_multiple_accumulations(self):
        total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        accumulate_usage(total, {"input_tokens": 100, "output_tokens": 50})
        accumulate_usage(total, {"input_tokens": 200, "output_tokens": 75})
        assert total["input_tokens"] == 300
        assert total["output_tokens"] == 125

    def test_handles_missing_keys(self):
        total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        accumulate_usage(total, {})  # Empty usage
        assert total == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
        }


class TestProcessToolCalls:
    """Tests for tool call processing."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        async def executor(name, args):
            return f"executed {name} with {args}"

        tool_calls = [
            {"id": "call_1", "function": {"name": "test_tool", "arguments": '{"x": 1}'}}
        ]
        results = await process_tool_calls(tool_calls, executor)

        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_1"
        assert "executed test_tool" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_parallel(self):
        call_order = []

        async def executor(name, args):
            call_order.append(name)
            await asyncio.sleep(0.01)  # Simulate work
            return f"done: {name}"

        tool_calls = [
            {"id": "call_1", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "tool_b", "arguments": "{}"}},
            {"id": "call_3", "function": {"name": "tool_c", "arguments": "{}"}},
        ]
        results = await process_tool_calls(tool_calls, executor, parallel=True)

        assert len(results) == 3
        # All should be called (order may vary due to parallel execution)
        assert set(call_order) == {"tool_a", "tool_b", "tool_c"}

    @pytest.mark.asyncio
    async def test_tool_exception_converted_to_error_string(self):
        async def executor(name, args):
            raise ValueError("Something went wrong")

        tool_calls = [
            {"id": "call_1", "function": {"name": "bad_tool", "arguments": "{}"}}
        ]
        results = await process_tool_calls(tool_calls, executor)

        assert len(results) == 1
        assert "Tool error" in results[0]["content"]
        assert "Something went wrong" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_invalid_json_args(self):
        async def executor(name, args):
            return f"args: {args}"

        tool_calls = [
            {"id": "call_1", "function": {"name": "tool", "arguments": "invalid json"}}
        ]
        results = await process_tool_calls(tool_calls, executor)

        assert len(results) == 1
        assert "Error parsing arguments" in results[0]["content"]


class TestRunAgentLoop:
    """Tests for the main agent loop."""

    @pytest.mark.asyncio
    async def test_simple_completion_no_tools(self):
        """Agent returns immediately without tool calls."""
        call_count = [0]

        async def call_model(messages, tools):
            call_count[0] += 1
            return (
                {"role": "assistant", "content": "Hello!", "tool_calls": []},
                {"input_tokens": 10, "output_tokens": 5},
            )

        async def executor(name, args):
            return "not called"

        result = await run_agent_loop(
            call_model=call_model,
            tool_executor=executor,
            initial_messages=[{"role": "user", "content": "hi"}],
        )

        assert result.success is True
        assert result.final_message == "Hello!"
        assert result.iterations == 1
        assert call_count[0] == 1
        assert result.usage["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_tool_call_then_completion(self):
        """Agent calls a tool, then completes."""
        iteration = [0]

        async def call_model(messages, tools):
            iteration[0] += 1
            if iteration[0] == 1:
                # First call: request tool
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "get_time", "arguments": "{}"},
                            }
                        ],
                    },
                    {"input_tokens": 10},
                )
            else:
                # Second call: complete
                return (
                    {
                        "role": "assistant",
                        "content": "The time is 12:00",
                        "tool_calls": [],
                    },
                    {"input_tokens": 20},
                )

        async def executor(name, args):
            return "12:00"

        result = await run_agent_loop(
            call_model=call_model,
            tool_executor=executor,
            initial_messages=[{"role": "user", "content": "what time is it"}],
        )

        assert result.success is True
        assert result.final_message == "The time is 12:00"
        assert result.iterations == 2
        assert result.tool_calls_count == 1
        assert result.usage["input_tokens"] == 30

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        """Loop stops at max iterations."""

        async def call_model(messages, tools):
            # Always request a tool call
            return (
                {
                    "role": "assistant",
                    "content": "thinking...",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "loop", "arguments": "{}"}}
                    ],
                },
                {},
            )

        async def executor(name, args):
            return "looping"

        result = await run_agent_loop(
            call_model=call_model,
            tool_executor=executor,
            initial_messages=[{"role": "user", "content": "loop forever"}],
            max_iterations=3,
        )

        assert result.success is False
        assert result.iterations == 3
        # Should return last assistant content when max iterations reached
        assert result.final_message == "thinking..."

    @pytest.mark.asyncio
    async def test_before_tool_execution_hook_can_discard(self):
        """before_tool_execution returning False discards tool calls."""
        iteration = [0]
        discard_first = [True]

        async def call_model(messages, tools):
            iteration[0] += 1
            if iteration[0] <= 2:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "t", "arguments": "{}"}}
                        ],
                    },
                    {},
                )
            else:
                return ({"role": "assistant", "content": "done", "tool_calls": []}, {})

        async def executor(name, args):
            return "executed"

        async def before_tools(assistant_message):
            if discard_first[0]:
                discard_first[0] = False
                return False  # Discard first tool call
            return True

        hooks = LoopHooks(before_tool_execution=before_tools)

        result = await run_agent_loop(
            call_model=call_model,
            tool_executor=executor,
            initial_messages=[],
            hooks=hooks,
        )

        assert result.success is True
        # iteration 1: tool call discarded, iteration 2: tool executed, iteration 3: done
        assert result.iterations == 3
        assert result.tool_calls_count == 1  # Only one actually executed

    @pytest.mark.asyncio
    async def test_before_final_response_can_discard(self):
        """before_final_response returning False retries final response."""
        iteration = [0]
        discard_first = [True]

        async def call_model(messages, tools):
            iteration[0] += 1
            return (
                {
                    "role": "assistant",
                    "content": f"done {iteration[0]}",
                    "tool_calls": [],
                },
                {},
            )

        async def executor(name, args):
            return "not used"

        async def before_final(assistant_message):
            if discard_first[0]:
                discard_first[0] = False
                return False
            return True

        hooks = LoopHooks(before_final_response=before_final)

        result = await run_agent_loop(
            call_model=call_model,
            tool_executor=executor,
            initial_messages=[],
            hooks=hooks,
        )

        assert result.success is True
        assert result.iterations == 2
        assert result.final_message == "done 2"

    @pytest.mark.asyncio
    async def test_hooks_called_in_order(self):
        """All hooks are called in the expected order."""
        events = []

        async def call_model(messages, tools):
            events.append("model_call")
            return ({"role": "assistant", "content": "done", "tool_calls": []}, {})

        async def executor(name, args):
            return "x"

        async def before_model(messages):
            events.append("before_model")
            return messages, False

        async def after_model(iteration, assistant_message, usage):
            events.append("after_model")

        async def on_final(success, iterations, final_message, total_usage):
            events.append("on_final")

        hooks = LoopHooks(
            before_model_call=before_model,
            after_model_call=after_model,
            on_final=on_final,
        )

        await run_agent_loop(
            call_model=call_model,
            tool_executor=executor,
            initial_messages=[],
            hooks=hooks,
        )

        assert events == ["before_model", "model_call", "after_model", "on_final"]
