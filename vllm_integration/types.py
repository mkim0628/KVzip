# ------------------------------------------------------------------------------
# vLLM-compatible types for KVzip integration.
# These mirror vLLM's SamplingParams, CompletionOutput, and RequestOutput
# so that KVzipVLLMEngine can be used as a drop-in replacement.
# ------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SamplingParams:
    """
    Sampling parameters for text generation.
    Compatible with vLLM's SamplingParams interface.
    """
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1            # -1 = disabled
    max_tokens: int = 512
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[List[str]] = None
    n: int = 1                 # number of output sequences

    def __post_init__(self):
        assert 0.0 <= self.temperature, "temperature must be non-negative"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.max_tokens > 0, "max_tokens must be positive"

    def to_hf_kwargs(self) -> Dict[str, Any]:
        """Convert to HuggingFace generate() kwargs."""
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.temperature == 0.0:
            kwargs["do_sample"] = False
            kwargs["temperature"] = 1.0
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
        if self.top_k > 0:
            kwargs["top_k"] = self.top_k
        return kwargs


@dataclass
class CompletionOutput:
    """
    A single completion output.
    Compatible with vLLM's CompletionOutput interface.
    """
    index: int
    text: str
    finish_reason: str = "stop"    # "stop" | "length" | "error"
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None


@dataclass
class RequestOutput:
    """
    The output of a generation request.
    Compatible with vLLM's RequestOutput interface.
    """
    request_id: str
    prompt: str
    outputs: List[CompletionOutput]
    prompt_token_ids: Optional[List[int]] = None
    finished: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Convenience accessor for the first output text."""
        return self.outputs[0].text if self.outputs else ""
