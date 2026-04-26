import json
import datetime
import os


class LLMCallLogger:
    def __init__(self, log_file: str = "llm_calls.json"):
        self.log_file = log_file
        self._call_counter = 0
        self._total_tokens = 0

    def log(
        self,
        call_type: str,
        context: str,
        prompt: str,
        raw_response: str,
        parsed_response=None,
        tool_name: str = None,
        tool_args: dict = None,
        tool_result: str = None,
        usage: dict = None,
        error: str = None,
    ):
        self._call_counter += 1
        if usage and "total_tokens" in usage:
            self._total_tokens += usage["total_tokens"]

        entry = {
            "call_id": self._call_counter,
            "timestamp": datetime.datetime.now().isoformat(),
            "call_type": call_type,
            "context": context,
            "prompt_sent": prompt,
            "raw_response": raw_response,
            "parsed_response": parsed_response,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result,
            "usage": usage,
            "error": error,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def summary(self) -> str:
        return (
            f"LLM CALLS LOGGED: {self.log_file} "
            f"({self._call_counter} calls, ~{self._total_tokens:,} tokens)"
        )

    def clear(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self._call_counter = 0
        self._total_tokens = 0
