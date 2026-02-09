import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass

from ..types import PreProcessResult, SequenceId, TokenConstraint, TokenId, TokenSplice


class CustomController:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self._initialized = False
        self._initialized = False
        self._id_to_future: dict[SequenceId, asyncio.Future] = {}
        self._event_loop = self._get_event_loop()
        self._id_to_string: dict[SequenceId, str] = {}

    def _get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if "There is no current event loop in thread" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        return loop

    def init(
        self, seq_id: SequenceId, prompt_tokens: Sequence[TokenId]
    ) -> Sequence[TokenId]:
        self._initialized = True
        return prompt_tokens

    def pre_process(self, seq_id: SequenceId) -> PreProcessResult:
        """Decide what to do with a sequence before token generation.

        Args:
            seq_id: Sequence identifier

        Returns:
            PreProcessResult indicating fork/continue/stop action
        """

        f = self._id_to_future.get(seq_id, None)
        if f:
            ff_tokens = self._event_loop.run_until_complete(f)
            return PreProcessResult(action="continue", extra_tokens=ff_tokens)

        # Continue active sequences
        return PreProcessResult(action="continue")

    def mid_process(self, seq_id: SequenceId) -> TokenConstraint:  # noqa: ARG002
        """Apply token constraints before sampling.

        Beam search uses greedy/sampling without hard constraints.

        Args:
            seq_id: Sequence identifier

        Returns:
            TokenConstraint (unconstrained for basic beam search)
        """
        # Beam search doesn't constrain tokens - just samples and prunes later
        return TokenConstraint.unconstrained()

    def post_fork(self, parent_id: SequenceId, child_ids: Sequence[SequenceId]) -> None:
        pass

    def post_process(
        self,
        seq_id: SequenceId,
        token: int,
        logprob: float,
        correction: float = 0.0,  # noqa: ARG002
    ) -> TokenSplice:

        if seq_id not in self._id_to_string:
            self._id_to_string[seq_id] = ""

        s = self.tokenizer.decode([token])
        self._id_to_string[seq_id] += s

        f = asyncio.ensure_future(self._process_string(seq_id), loop=self._event_loop)
        self._id_to_future[seq_id] = f

        return TokenSplice(backtrack=0, ff_tokens=(token,))

    async def _process_string(self, seq_id: SequenceId) -> list[TokenId]:
        text = self._id_to_string[seq_id]
        if "<calling_tool>" in text and "</calling_tool>" in text:
            # forced_text = """\nTOOL_RESULT: 841265283915631 + 285623858 = 841265569539489"""
            # _tokens = self.tokenizer(forced_text, add_special_tokens=False).input_ids
            # self._id_to_string[seq_id] = ""

            start_idx = text.index("<calling_tool>") + len("<calling_tool>")
            end_idx = text.index("</calling_tool>")
            query_str = text[start_idx:end_idx].strip()

            query = json.loads(query_str)
            if query["name"] == "Add":
                a = query["args"]["a"]
                b = query["args"]["b"]
                result = f"{a} + {b} = {a + b}"
            elif query["name"] == "Sub":
                a = query["args"]["a"]
                b = query["args"]["b"]
                result = f"{a} - {b} = {a - b}"
            else:
                result = f"Unknown tool {query['name']}"

            forced_text = f"\nTOOL_RESULT: {result}"
            _tokens = self.tokenizer(forced_text, add_special_tokens=False).input_ids
            self._id_to_string[seq_id] = ""

            return _tokens

        return []


__all__ = ["CustomController"]
