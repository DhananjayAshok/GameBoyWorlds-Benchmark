import httpx
from openai import APITimeoutError, AsyncOpenAI
from anthropic import AsyncAnthropic
from time import sleep, perf_counter
import asyncio
import random
from typing import Optional, Any, Union
from utils.parameter_handling import load_parameters
from utils.log_handling import log_info, log_warn, log_error
import shutil
from PIL import Image
import base64
import os
from abc import ABC, abstractmethod
import uuid


def parse_key_value(text: str, key: str) -> Optional[str]:
    """
    Return the value following ``"Key:"`` on the matching line of ``text``.

    The key is matched case-insensitively; the returned value preserves its
    original case.

    If ``text`` contains exactly one ``"response:"`` occurrence, only the text
    after it is searched (avoids matching mentions of ``key`` in a preceding
    "Reasoning:" section). If no ``"key:"`` line is found but ``key`` (without
    a colon) appears exactly once, the rest of that occurrence's line is
    returned instead.

    :param text: The text to search.
    :type text: str
    :param key: The key to search for.
    :type key: str
    :return: The extracted value, or None if not found or empty.
    :rtype: Optional[str]
    """
    key_lower = key.lower()
    marker = f"{key_lower}:"

    def clean(value: str) -> Optional[str]:
        value = value.strip()
        return value or None

    text_lower = text.lower()
    if text_lower.count("response:") == 1:  # sometimes API models do this.
        idx = text_lower.index("response:") + len("response:")
        text = text[idx:].strip()
        text_lower = text.lower()

    for line, line_lower in zip(text.splitlines(), text_lower.splitlines()):
        idx = line_lower.find(marker)
        if idx != -1:
            return clean(line[idx + len(marker) :])

    if text_lower.count(key_lower) == 1:
        idx = text_lower.index(key_lower)
        rest_of_line = text[idx + len(key) :].splitlines()
        return clean(rest_of_line[0]) if rest_of_line else None

    return None


def clean_value(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip().strip('"\'').strip()
    if not value or value.lower() in ("none", "n/a", "na", "unknown"):
        return None
    return value


def parse_yes_no(text: str, key: str) -> Optional[bool]:
    """
    Return the yes/no verdict on the ``"Key:"`` line of ``text``.

    :param text: The model response to search.
    :type text: str
    :param key: The verdict key, e.g. ``"Complete"``, ``"Relevant"``, ``"Flawed"``.
    :type key: str
    :return: ``True`` on an explicit yes, ``False`` on any other answer, ``None``
        when no answer was given.
    :rtype: Optional[bool]
    """
    raw = parse_key_value(text, key)
    if raw is None:
        return None
    return raw.strip().lower().startswith("yes")


MIN_QUERIES_PER_MINUTE = 1

#: Retries for a timed-out call to a hosted API. A timeout means the provider stopped
#: answering, not that we asked too fast, so the rate-limit-derived backoff below is far
#: too short to outlast it: a 200 queries/minute model waits 0.3s, and five of those are
#: spent inside a few minutes of a stall that can last much longer. A run is hours long
#: and its state is checkpointed per episode, so waiting is nearly always better than
#: dying. Local vLLM is excluded: a hung server there will not heal on its own.
TIMEOUT_MAX_TRIES = 20
TIMEOUT_BACKOFF_FLOOR = 60.0
TIMEOUT_BACKOFF_CAP = 600.0


def _is_timeout(error: Exception) -> bool:
    return isinstance(error, (APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError))


def _retry_backoff(attempt: int, seconds_to_wait: float, timeout_retry: bool) -> float:
    """Seconds to wait before the retry following ``attempt`` (0-based), with jitter.

    Jitter spreads coroutines that all failed together, which matters most here: one
    perception batch can be tens of concurrent calls that time out in the same second.
    """
    if timeout_retry:
        backoff = TIMEOUT_BACKOFF_FLOOR * (2**attempt) * random.uniform(1.0, 1.5)
        return min(backoff, TIMEOUT_BACKOFF_CAP)
    return seconds_to_wait * (2**attempt) * random.uniform(1.0, 1.5)


# Placeholder per-model rate limits (queries per minute).
_RATE_LIMITS: dict[str, int] = {
    "gpt-4o-mini": 60,
    "gpt-4o": 60,
    "gpt-4": 60,
    "gpt-5": 60,
    "claude-opus-4.7": 60,
    "claude-sonnet-4-6": 60,
    "claude-haiku-4-5-20251001": 60,
    "google/gemini-3.1-pro-preview": 60,
    # Raised above the default: Flash is the cheap, high-throughput arm.
    "google/gemini-3.6-flash": 200,
    "google/gemini-4-argon": 120,
    "qwen/qwen3-vl-235b-a22b-instruct": 60,
    "google/gemma-4-31b-it": 1000,
}


# Model names for which the "not found in _RATE_LIMITS" warning has already
# been logged once. Avoids re-logging the same warning every time a model
# with no explicit rate limit is instantiated (e.g. once per attempt episode).
_WARNED_MISSING_RATE_LIMIT: set[str] = set()


def get_max_queries_per_minute(model: str, parameters: dict[str, Any]) -> int:
    """
    Look up the per-model rate limit (queries per minute) from ``_RATE_LIMITS``.

    A key matches ``model`` if the key equals ``model``, the key is a substring
    of ``model``, or ``model`` is a substring of the key. If multiple keys
    match, the longest (most specific) one wins. Falls back to
    ``parameters["default_max_queries_per_minute"]`` (logging a warning, once
    per model name) if no key matches.

    :param model: The model identifier string.
    :type model: str
    :param parameters: Loaded parameters dict.
    :type parameters: dict[str, Any]
    :return: The queries-per-minute limit to use for ``model``.
    :rtype: int
    """
    matches = [key for key in _RATE_LIMITS if key in model or model in key]
    if not matches:
        if model not in _WARNED_MISSING_RATE_LIMIT:
            log_warn(
                f"Model {model} not found in _RATE_LIMITS (no exact or substring match). "
                f"Using default_max_queries_per_minute from project parameters.",
                parameters=parameters,
            )
            _WARNED_MISSING_RATE_LIMIT.add(model)
        return parameters["default_max_queries_per_minute"]
    else:
        if len(matches) > 1:
            for match in matches:
                if match == model.split("/")[-1]:
                    return _RATE_LIMITS[match]
            log_error(
                f"Multiple matches found in _RATE_LIMITS for model {model}: {matches}. "
                f"Please disambiguate by adding a more specific key to _RATE_LIMITS.",
                parameters=parameters,
            )
        else:
            return _RATE_LIMITS[matches[0]]


def sum_optional(values: list[Optional[int]]) -> Optional[int]:
    """
    Sum token counts, propagating unknowns. ``None`` means "not reported by the backend" and
    makes the whole sum ``None``.

    :param values: Token counts, any of which may be None.
    :type values: list[Optional[int]]
    :return: The total, or None if any value was None.
    :rtype: Optional[int]
    """
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def sum_meta(*metas: dict[str, Any]) -> dict[str, Optional[int]]:
    """
    Add ``meta`` dicts together, flattening list-valued entries.

    :param metas: Any number of ``{"input_tokens": ..., "output_tokens": ...}`` dicts,
        whose values may be ints, None, or lists of either.
    :return: One ``{"input_tokens": ..., "output_tokens": ...}`` with scalar values.
    :rtype: dict[str, Optional[int]]
    """
    totals = {}
    for key in ("input_tokens", "output_tokens"):
        flat: list[Optional[int]] = []
        for meta in metas:
            value = meta.get(key) if meta else None
            if isinstance(value, list):
                flat.extend(value)
            else:
                flat.append(value)
        totals[key] = sum_optional(flat)
    return totals


def zero_meta() -> dict[str, int]:
    """
    The ``meta`` of a code path that made no inference calls at all.

    :return: ``{"input_tokens": 0, "output_tokens": 0}``
    :rtype: dict[str, int]
    """
    return {"input_tokens": 0, "output_tokens": 0}


def _collapse_meta(meta: dict[str, list[Optional[int]]]) -> dict[str, Optional[int]]:
    """
    Collapse a single-record meta dict (each value a length-1 list) to scalar values.

    :param meta: Meta dict whose values are lists of length 1.
    :type meta: dict[str, list[Optional[int]]]
    :return: The same dict with each value replaced by its single element.
    :rtype: dict[str, Optional[int]]
    """
    return {key: value[0] for key, value in meta.items()}


def _extract_usage(
    response: Any,
    *,
    input_attr: str,
    output_attr: str,
    parameters: dict[str, Any] = None,
) -> tuple[Optional[int], Optional[int]]:
    """
    Read the token usage off an API response, tolerating providers that omit it.

    :param response: The raw response object returned by the API client.
    :type response: Any
    :param input_attr: Name of the input-token attribute on ``response.usage``
        (``"prompt_tokens"`` for OpenAI-compatible, ``"input_tokens"`` for Anthropic).
    :type input_attr: str
    :param output_attr: Name of the output-token attribute on ``response.usage``.
    :type output_attr: str
    :param parameters: Loaded parameters dict, used for logging.
    :type parameters: dict[str, Any] or None
    :return: ``(input_tokens, output_tokens)``, either of which is None if the response
        did not report it.
    :rtype: tuple[Optional[int], Optional[int]]
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        log_warn(
            "API response did not report token usage; recording input/output tokens as None.",
            parameters=parameters,
        )
        return None, None
    input_tokens = getattr(usage, input_attr, None)
    output_tokens = getattr(usage, output_attr, None)
    if input_tokens is None or output_tokens is None:
        log_warn(
            f"API response usage is missing {input_attr}/{output_attr} "
            f"(got {input_tokens}/{output_tokens}); recording the missing count as None.",
            parameters=parameters,
        )
    return input_tokens, output_tokens


class RateLimitedAPIBase:
    """
    Mixin that provides rate-limited API client state and ``wait()`` logic.

    Shared by ``APIModel`` and any future rate-limited API-backed model
    wrapper to avoid duplicating the init and rate-limiting code.
    """

    def __init__(
        self,
        *,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        self.parameters = load_parameters(parameters)
        self.model = model
        if max_queries_per_minute is None:
            max_queries_per_minute = get_max_queries_per_minute(model, self.parameters)
        self.max_queries_per_minute = max_queries_per_minute
        self.last_query_time = 0
        self.seconds_to_wait = 60 / self.max_queries_per_minute
        self.unique_id = str(uuid.uuid4())
        if 0 <= self.max_queries_per_minute < MIN_QUERIES_PER_MINUTE:
            log_error(
                f"max_queries_per_minute must be at least {MIN_QUERIES_PER_MINUTE}, "
                f"but got {self.max_queries_per_minute}.",
                parameters=self.parameters,
            )

    def wait(self) -> None:
        """
        Enforce the rate limit by sleeping until enough time has elapsed since the last query.

        Updates ``last_query_time`` after waiting.
        """
        time_to_wait = self.seconds_to_wait - (perf_counter() - self.last_query_time)
        if time_to_wait > 0:
            sleep(time_to_wait)
        self.last_query_time = perf_counter()


class OpenAICompatibleAPIBase(RateLimitedAPIBase):
    """
    Mixin that extends ``RateLimitedAPIBase`` with an OpenAI-compatible async client.

    Stores the ``AsyncOpenAI(base_url=..., api_key=...)`` constructor arguments
    after the rate-limiting state is set up. The client itself is created fresh
    by :meth:`_make_async_client` inside each ``asyncio.run()`` call (see
    ``APIModel._infer_messages_async``/``_do_infer_async``), so its connection
    pool is never reused across event loops. Shared by all OpenAI-compatible
    models (``OpenAIAPIModel`` and its subclasses) to avoid repeating client
    creation in every subclass.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self._async_client_base_url = base_url
        self._async_client_api_key = api_key

    def _make_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self._async_client_base_url,
            api_key=self._async_client_api_key,
            timeout=httpx.Timeout(600.0, connect=30.0),
        )


class InferenceModel(ABC):
    """
    Abstract base class for all LM inference that support inference
    """

    @abstractmethod
    def do_infer(
        self,
        texts: list[str],
        images: list[list[Image.Image]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """
        Run inference on a batch of text prompts with associated images. Assumes validated inputs

        ``images`` is positional and precedes ``max_new_tokens``: implementations
        are called from :meth:`infer` and must keep this order.

        :param texts: List of text prompts, one per sample.
        :type texts: list[str]
        :param images: List of image lists, one image list per sample.
        :type images: list[list[Image.Image]]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return per prompt.
        :type num_return_sequences: int
        :return: ``{"output": ..., "meta": ...}`` where ``output`` holds the post-processed
            output strings shaped ``[batch, num_return_sequences]`` and ``meta`` is
            ``{"input_tokens": [...], "output_tokens": [...]}`` with one entry per record
            (i.e. lists of length ``batch``), each entry an int or None if the backend did
            not report the count. See :meth:`_build_meta` for the per-record accounting.
        :rtype: dict[str, Any]
        """
        pass

    def _build_meta(
        self, *, usages: list[list[tuple[Optional[int], Optional[int]]]]
    ) -> dict[str, list[Optional[int]]]:
        """
        Aggregate per-sequence token counts into the per-record ``meta`` dict.

        ``meta`` is always per *record*: it never gains a ``num_return_sequences``
        dimension. A record's counts are the sum over its sequences, so accounting for
        ``num_return_sequences > 1`` differs by backend:

        - ``AnthropicModel``/``OpenRouterModel`` issue one call per sequence, so the prompt
          is consumed ``num_return_sequences`` times and is summed.
        - ``OpenAIAPIModel``/``vLLMModel`` use the API's native ``n``; the extra sequences
          carry ``0`` input tokens.
        - ``HuggingFaceModel`` encodes the prompt once per record, likewise.

        ``None`` means "not reported by the backend" and propagates.

        :param usages: Per-sequence ``(input_tokens, output_tokens)`` tuples shaped
            ``[batch, num_return_sequences]``.
        :type usages: list[list[tuple[Optional[int], Optional[int]]]]
        :return: ``{"input_tokens": [...], "output_tokens": [...]}``, each a list of
            length ``batch``.
        :rtype: dict[str, list[Optional[int]]]
        """
        return {
            "input_tokens": [
                sum_optional([usage[0] for usage in record_usages])
                for record_usages in usages
            ],
            "output_tokens": [
                sum_optional([usage[1] for usage in record_usages])
                for record_usages in usages
            ],
        }

    def get_output_final(self, output_text: str) -> str:
        """
        Post-process a single output text by truncating at the ``[STOP]`` token (case-insensitive) and stripping whitespace.

        :param output_text: Raw output string from the model.
        :type output_text: str
        :return: Cleaned output string with content after ``[STOP]`` removed.
        :rtype: str
        """
        stop_idx = output_text.lower().find("[stop]")
        if stop_idx != -1:
            output_text = output_text[:stop_idx]
        return output_text.strip()

    def _standardize_format(
        self,
        texts: Union[str, list[str]],
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
    ) -> tuple[list[str], list[list[Image.Image]], bool]:
        """
        Validates and standardizes the format of ``texts`` and ``images`` inputs as per do_infer's expectations.

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
        :param images: A list of PIL Images (when ``texts`` is a single string) or a list of lists
            of PIL Images (when ``texts`` is a list). If None, no images are passed.
        :type images: list[Image.Image] or list[list[Image.Image]] or None
        :return: A tuple of (standardized_texts, standardized_images, passed_in_str) where standardized_texts is a list of strings and standardized_images is a list of lists of PIL Images both formatted for input to do_infer. passed_in_str is a boolean indicating whether the original input was a single string (True) or a list of strings (False), which can be used to determine the appropriate output format in infer().
        :rtype: tuple[list[str], list[list[Image.Image]], bool]
        """
        passed_in_str = isinstance(texts, str)
        if passed_in_str:
            if texts.strip() == "":
                log_error(f"texts cannot be empty")
            texts = [texts]
        else:
            if not isinstance(texts, list):
                log_error(
                    f"texts must be a string or list of strings. Got {type(texts)}"
                )
            if len(texts) == 0:
                log_error(f"texts cannot be empty.")
            for item in texts:
                if not isinstance(item, str):
                    log_error(f"Got {type(item)}:{item} instead of str as text")

        if images is not None:
            if not isinstance(images, list):
                log_error(
                    f"images must be a list of PIL images or list of lists of PIL Images. Got {type(images)}"
                )
            else:
                if len(images) == 0:
                    log_error(f"images cannot be empty")
                if passed_in_str:
                    for item in images:
                        if not isinstance(item, Image.Image):
                            if isinstance(item, list):
                                log_error(
                                    f"Passed in a single string for texts but  list of lists for images. This is confusing."
                                )
                            log_error(
                                f"image list contains non images: {type(item)}: {item}"
                            )
                    images = [images]
                else:
                    for list_item in images:
                        if not isinstance(list_item, list):
                            log_error(
                                f"images must be a list of list of PIL Images, got a {type(list_item)}:{list_item}"
                            )
                        for item in list_item:
                            if not isinstance(item, Image.Image):
                                log_error(
                                    f"image list contains non images: {type(item)}: {item}"
                                )
            if len(texts) != len(images):
                log_error(
                    f"Number of text prompts and number of image lists must be the same. Got {len(texts)} text prompts and {len(images)} image lists."
                )
        else:
            images = [[] for _ in texts]
        return texts, images, passed_in_str

    def infer(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
        batch_size: int = None,
    ) -> dict[str, Any]:
        """
        Run inference on a batch of text prompts with associated images.

        Returns ``{"output": ..., "meta": ...}``.

        ``output`` follows the input shape: if a single string is passed, a single string
        is returned; if a list is passed, a list is returned. When
        ``num_return_sequences > 1``, each item is itself a list of
        ``num_return_sequences`` output strings.

        ``meta`` is ``{"input_tokens": ..., "output_tokens": ...}`` with one entry **per
        record** — a bare int (or None) if a single string was passed, otherwise a list of
        length ``len(texts)``. ``meta`` never gains a ``num_return_sequences`` dimension;
        a record's output tokens are summed over its sequences. See :meth:`_build_meta`
        for how each backend accounts for the prompt when ``num_return_sequences > 1``.

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param images: A list of PIL Images (when ``texts`` is a single string) or a list of lists
            of PIL Images (when ``texts`` is a list). If None, no images are passed.
        :type images: list[Image.Image] or list[list[Image.Image]] or None
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return per prompt.
        :type num_return_sequences: int
        :param batch_size: Number of samples to process in a single batch. If None, defaults to
            ``max_batch_size_vllm``, ``max_batch_size_huggingface``, or ``max_batch_size_api`` from
            project parameters, depending on the concrete model class.
        :return: ``{"output": ..., "meta": ...}``. ``output`` is a single output string if
            ``texts`` was a string and ``num_return_sequences == 1``; a list of output
            strings if ``texts`` was a list and ``num_return_sequences == 1``; a list of
            ``num_return_sequences`` strings if ``texts`` was a string and
            ``num_return_sequences > 1``; or a list of such lists otherwise. ``meta`` holds
            per-record ``input_tokens``/``output_tokens``, scalars if ``texts`` was a string
            and lists of length ``len(texts)`` otherwise.
        :rtype: dict[str, Any]
        """
        texts, images, passed_in_str = self._standardize_format(texts, images)
        parameters = (
            self.parameters if hasattr(self, "parameters") else load_parameters()
        )
        if batch_size is None:
            from utils.huggingface_inference import HuggingFaceModel

            if isinstance(self, vLLMModel):
                batch_size = parameters["max_batch_size_vllm"]
            elif isinstance(self, HuggingFaceModel):
                batch_size = parameters["max_batch_size_huggingface"]
            else:
                batch_size = parameters["max_batch_size_api"]
        results = []
        meta = {"input_tokens": [], "output_tokens": []}
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_images = images[i : i + batch_size]
            batch_results = self.do_infer(
                texts=batch_texts,
                images=batch_images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_strings=stop_strings,
                num_return_sequences=num_return_sequences,
            )
            results.extend(batch_results["output"])
            for key in meta:
                meta[key].extend(batch_results["meta"][key])
        if passed_in_str:
            meta = _collapse_meta(meta)
        if num_return_sequences == 1:
            if passed_in_str:
                output = results[0][0]
            else:
                output = [r[0] for r in results]
        else:
            if passed_in_str:
                output = results[0]
            else:
                output = results
        return {"output": output, "meta": meta}

    @abstractmethod
    def infer_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """
        Run inference on a pre-formatted chat messages list.

        :return: ``{"output": ..., "meta": ...}``. ``output`` is a single output string if
            ``num_return_sequences == 1``, else a list of ``num_return_sequences`` output
            strings. A messages list is a single record, so ``meta``'s
            ``input_tokens``/``output_tokens`` are scalars (int or None) regardless of
            ``num_return_sequences``, with output tokens summed over the sequences.
        :rtype: dict[str, Any]
        """
        pass

    def partial_temperature(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        switch_phrase: str,
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """
        Run inference once at ``temperature``, then deterministically complete the
        portion of the output following ``switch_phrase``.

        Samples a "thinking" prefix at ``temperature``; if ``switch_phrase`` is found
        in the output, truncates before it and re-queries (at ``temperature=None``,
        i.e. deterministic) with the truncated output plus ``switch_phrase`` appended
        to the prompt, to obtain a deterministic final answer.

        .. note:: **``meta`` here is a scalar total, not per-record**, covering both passes
            for the whole call. This function has no callers in this repo; make the nesting
            of ``first_outputs_list`` consistent before relying on it.

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param switch_phrase: Phrase marking the boundary between "thinking" and "answer".
        :type switch_phrase: str
        :param images: A list of PIL Images, or None.
        :type images: list[Image.Image] or None
        :param temperature: Sampling temperature for the first pass. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return.
        :type num_return_sequences: int
        :return: ``{"output": ..., "meta": ...}``. ``output`` is the completed output string
            (or None if ``switch_phrase`` was not found), or a list of such results when
            ``num_return_sequences > 1``. ``meta`` is a scalar
            ``{"input_tokens": ..., "output_tokens": ...}`` covering both passes — see the
            note above.
        :rtype: dict[str, Any]
        """
        texts, images, passed_in_str = self._standardize_format(texts, images)
        asked_for_single_sequence = num_return_sequences == 1
        first_result = self.infer(
            texts,
            max_new_tokens,
            images=images,
            temperature=temperature,
            stop_strings=stop_strings,
            num_return_sequences=num_return_sequences,
        )
        first_outputs = first_result["output"]

        first_outputs_list = (
            [first_outputs] if asked_for_single_sequence else first_outputs
        )  # always a doubly nested list.
        next_batch_text = []
        next_batch_images = []
        next_batch_output_so_fars = []
        next_batch_mapping = {}
        largest_max_tokens = 5
        for og_batch_i, batch_outputs in enumerate(first_outputs_list):
            for return_seq_i, output in enumerate(batch_outputs):
                if switch_phrase not in output:
                    continue
                output_so_far = output.split(switch_phrase)[0]
                n_tokens_estimated = len(output_so_far.split())
                largest_max_tokens = max(
                    largest_max_tokens, max_new_tokens - n_tokens_estimated
                )
                second_prompt = (
                    texts[og_batch_i]
                    + "\nHere is what you said: "
                    + output_so_far
                    + "\n"
                    + switch_phrase
                    + " "
                )
                next_batch_idx = len(next_batch_text)
                next_batch_mapping[(og_batch_i, return_seq_i)] = next_batch_idx
                next_batch_text.append(second_prompt)
                next_batch_images.append(images[og_batch_i])
                next_batch_output_so_fars.append(output_so_far)

        if len(next_batch_text) == 0:
            # The first pass still cost tokens even though no second pass was needed.
            meta = sum_meta(first_result["meta"])
            if asked_for_single_sequence:
                if passed_in_str:
                    output = None
                else:
                    output = [None for _ in texts]
            else:
                if passed_in_str:
                    output = [None for _ in range(num_return_sequences)]
                else:
                    output = [
                        [None for _ in range(num_return_sequences)] for _ in texts
                    ]
            return {"output": output, "meta": meta}
        second_result = self.infer(
            next_batch_text,
            largest_max_tokens,
            images=next_batch_images,
            temperature=None,
            stop_strings=stop_strings,
            num_return_sequences=1,
        )
        second_output = second_result["output"]
        for i, output in enumerate(second_output):
            second_output[i] = (
                next_batch_output_so_fars[i]
                + "\n"
                + switch_phrase
                + " "
                + output.lstrip(switch_phrase).lstrip()
            )

        results = []
        for og_batch_i in range(len(first_outputs_list)):
            n_return_seqs = len(first_outputs_list[og_batch_i])
            batch_results = []
            for return_seq_i in range(n_return_seqs):
                if (og_batch_i, return_seq_i) in next_batch_mapping:
                    # breakpoint()
                    target_i = next_batch_mapping[(og_batch_i, return_seq_i)]
                    batch_results.append(second_output[target_i])
                else:
                    batch_results.append(None)
            results.append(batch_results)
        for i, result in enumerate(results):
            if passed_in_str:
                results[i] = result[0]
        meta = sum_meta(first_result["meta"], second_result["meta"])
        if asked_for_single_sequence:
            return {"output": results[0], "meta": meta}
        return {"output": results, "meta": meta}


class APIModel(RateLimitedAPIBase, InferenceModel, ABC):
    """
    Abstract base class for API-backed language and vision-language models.

    Handles rate limiting, image encoding, and output post-processing.
    Subclasses must implement ``get_image_input_dict``, ``query_client``,
    and ``get_output_texts``.
    """

    SUPPORTS_NATIVE_N: bool = False

    #: Whether the endpoint is one we run ourselves (vLLM), where a timeout means a hung or
    #: dead server, not a briefly unreachable provider.
    LOCAL_ENDPOINT: bool = False

    def __init__(
        self,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the base API model with rate limiting and parameter loading.

        :param model: The model identifier string (e.g. ``"gpt-4o"``).
        :type model: str
        :param max_queries_per_minute: Maximum number of queries allowed per minute. Must be at least 1.
        :type max_queries_per_minute: Optional[int]
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    def get_encoded_images(self, images: list[Image.Image]) -> list[str]:
        """Encodes images to base64 strings for OpenAI API input, staging them in a fresh
        per-call cache directory.

        :param images: List of images in Pillow Image format.
        :type images: list[Image.Image]
        :return: List of base64 encoded image strings.
        :rtype: list[str]
        """
        from python_scripts.paths import api_image_cache_dir

        cache_dir = os.path.join(
            api_image_cache_dir(self.parameters, unique_id=self.unique_id),
            str(uuid.uuid4()),
        )
        os.makedirs(cache_dir)
        try:
            encoded_images = []
            for i, img in enumerate(images):
                img_path = os.path.join(cache_dir, f"image_{i}.jpg")
                if img.mode in ("RGBA", "P", "LA"):
                    img = img.convert("RGB")
                img.save(img_path, format="JPEG")
                with open(img_path, "rb") as image_file:
                    encoded_images.append(
                        base64.b64encode(image_file.read()).decode("utf-8")
                    )
        finally:
            shutil.rmtree(cache_dir)
        return encoded_images

    @abstractmethod
    def get_image_input_dict(self, image: str) -> dict:
        """
        Return the API-specific content dict for a single base64-encoded image.

        :param image: A base64-encoded image string.
        :type image: str
        :return: A dictionary formatted for inclusion in the API message content.
        :rtype: dict
        """
        pass

    @abstractmethod
    def _make_async_client(self) -> Any:
        """
        Construct a fresh async API client (e.g. ``AsyncOpenAI``/``AsyncAnthropic``).

        Called once per ``asyncio.run()`` invocation (see
        ``_infer_messages_async``/``_do_infer_async``) so the client's
        connection pool — and any event-loop-bound primitives it lazily
        creates — never outlives the loop it was created in.

        :return: A newly constructed async client, usable as an async context manager.
        :rtype: Any
        """
        pass

    @abstractmethod
    async def query_client(
        self,
        client: Any,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> Any:
        """
        Send messages to the API client (asynchronously) and return the raw response.

        :param messages: A list of message dicts formatted for the API.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of sequences to return per prompt, if natively supported.
        :type num_return_sequences: int
        :return: Response from API
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_output_texts(
        self, response: Any
    ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
        """
        Extract raw output text strings and token usage from a single model API response.

        The usage list is parallel to the text list. Where a response reports a single
        ``usage`` covering several choices (the native-``n`` case), the counts are
        emitted on the first entry and the remaining entries carry ``0`` — meaning
        "already counted on a sibling entry", so summing a record's entries yields the
        correct total. ``None`` means the API did not report the count at all.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: ``(texts, usages)`` where ``texts`` holds one output string per sequence
            in the response and ``usages`` holds the matching
            ``(input_tokens, output_tokens)`` tuples.
        :rtype: tuple[list[str], list[tuple[Optional[int], Optional[int]]]]
        """
        pass

    def get_outputs(
        self, response: Any
    ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
        """
        Extract and post-process all output texts from a single API response.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: ``(texts, usages)`` where ``texts`` holds the cleaned output strings, one
            per sequence, and ``usages`` holds the matching
            ``(input_tokens, output_tokens)`` tuples (passed through unchanged).
        :rtype: tuple[list[str], list[tuple[Optional[int], Optional[int]]]]
        """
        texts, usages = self.get_output_texts(response)
        return [self.get_output_final(t) for t in texts], usages

    def get_output(
        self, response: Any
    ) -> tuple[str, tuple[Optional[int], Optional[int]]]:
        """
        Extract and post-process the first output text from a single API response.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: ``(text, usage)`` for the first sequence, where ``usage`` is
            ``(input_tokens, output_tokens)``.
        :rtype: tuple[str, tuple[Optional[int], Optional[int]]]
        """
        texts, usages = self.get_outputs(response)
        return texts[0], usages[0]

    def infer_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """
        Run inference on a pre-formatted chat messages list via the API.

        :return: ``{"output": ..., "meta": ...}``. ``output`` is a single output string if
            ``num_return_sequences == 1``, else a list of ``num_return_sequences`` output
            strings. A messages list is a single record, so ``meta``'s
            ``input_tokens``/``output_tokens`` are scalars (int or None) regardless of
            ``num_return_sequences``, with output tokens summed over the sequences. When
            ``SUPPORTS_NATIVE_N`` is False the ``num_return_sequences`` separate calls each
            consume the prompt, so ``input_tokens`` is their sum; see :meth:`_build_meta`.
        :rtype: dict[str, Any]
        """
        if num_return_sequences > 1 and temperature is None:
            log_error(
                f"num_return_sequences={num_return_sequences} requires temperature to be set "
                f"(got temperature=None); otherwise all sequences would be identical.",
                parameters=self.parameters,
            )
        outputs, usages = asyncio.run(
            self._infer_messages_async(
                messages,
                max_new_tokens,
                temperature,
                stop_strings,
                num_return_sequences,
            )
        )
        meta = _collapse_meta(self._build_meta(usages=[usages]))
        if num_return_sequences == 1:
            return {"output": outputs[0], "meta": meta}
        return {"output": outputs, "meta": meta}

    async def _infer_messages_async(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float],
        stop_strings: list[str],
        num_return_sequences: int,
    ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
        """
        Issue the request(s) for a single chat messages list and return ``num_return_sequences``
        output strings alongside their matching ``(input_tokens, output_tokens)`` tuples.

        A single ``self.wait()`` paces this call relative to the last request issued;
        all ``num_return_sequences`` requests (if multiple) are then fired concurrently.
        Per-request rate-limit errors are handled by ``query_client``'s retry/backoff.
        """
        async with self._make_async_client() as client:
            self.wait()
            if self.SUPPORTS_NATIVE_N:
                response = await self.query_client(
                    client,
                    messages,
                    max_new_tokens,
                    temperature=temperature,
                    stop_strings=stop_strings,
                    num_return_sequences=num_return_sequences,
                )
                outputs, usages = self.get_outputs(response)
                if len(outputs) != num_return_sequences:
                    log_error(
                        f"Expected {num_return_sequences} outputs but got {len(outputs)}. Response was: {response}",
                        parameters=self.parameters,
                    )
                if len(usages) != len(outputs):
                    log_error(
                        f"Expected {len(outputs)} usage entries but got {len(usages)}. Response was: {response}",
                        parameters=self.parameters,
                    )
                return outputs, usages
            else:

                async def query_one() -> (
                    tuple[str, tuple[Optional[int], Optional[int]]]
                ):
                    response = await self.query_client(
                        client,
                        messages,
                        max_new_tokens,
                        temperature=temperature,
                        stop_strings=stop_strings,
                    )
                    return self.get_output(response)

                return list(
                    await asyncio.gather(
                        *(query_one() for _ in range(num_return_sequences))
                    )
                )

    def do_infer(
        self,
        texts: list[str],
        images: list[list[Image.Image]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """
        Encodes all images to base64, constructs API message dicts, enforces
        the rate limit, queries the client, and returns post-processed outputs.

        :param texts: List of text prompts, one per sample.
        :type texts: list[str]
        :param images: List of image lists, one image list per sample.
        :type images: list[list[Image.Image]]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return per prompt.
        :type num_return_sequences: int
        :return: ``{"output": ..., "meta": ...}`` where ``output`` holds the post-processed
            output strings shaped ``[batch, num_return_sequences]`` and ``meta`` holds
            per-record ``input_tokens``/``output_tokens`` lists of length ``batch``.
        :rtype: dict[str, Any]
        """
        if len(images[0]) != 0:
            all_images = []
            for img_list in images:
                all_images.append(self.get_encoded_images(img_list))
            images = all_images
        inputs = []
        for text, img_list in zip(texts, images):
            content = [{"type": "text", "text": text}]
            for img in img_list:
                content.append(self.get_image_input_dict(img))
            inputs.append({"role": "user", "content": content})

        if num_return_sequences > 1 and temperature is None:
            log_error(
                f"num_return_sequences={num_return_sequences} requires temperature to be set "
                f"(got temperature=None); otherwise all sequences would be identical.",
                parameters=self.parameters,
            )

        outputs, usages = asyncio.run(
            self._do_infer_async(
                inputs, max_new_tokens, temperature, stop_strings, num_return_sequences
            )
        )
        return {"output": outputs, "meta": self._build_meta(usages=usages)}

    async def _do_infer_async(
        self,
        inputs: list[dict],
        max_new_tokens: int,
        temperature: Optional[float],
        stop_strings: list[str],
        num_return_sequences: int,
    ) -> tuple[list[list[str]], list[list[tuple[Optional[int], Optional[int]]]]]:
        """
        Issue one query per input message concurrently and return outputs shaped
        ``[batch, num_return_sequences]``, alongside per-sequence
        ``(input_tokens, output_tokens)`` tuples nested identically.

        A single ``self.wait()`` paces the start of this batch relative to the last
        request issued; all requests within the batch are then fired concurrently.
        Per-request rate-limit errors are handled by ``query_client``'s retry/backoff.
        """
        async with self._make_async_client() as client:
            self.wait()
            if self.SUPPORTS_NATIVE_N:

                async def query_one(
                    input_message: dict,
                ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
                    response = await self.query_client(
                        client,
                        [input_message],
                        max_new_tokens,
                        temperature=temperature,
                        stop_strings=stop_strings,
                        num_return_sequences=num_return_sequences,
                    )
                    seq_outputs, seq_usages = self.get_outputs(response)
                    if len(seq_outputs) != num_return_sequences:
                        log_error(
                            f"Expected {num_return_sequences} outputs but got {len(seq_outputs)}. Response was: {response}",
                            parameters=self.parameters,
                        )
                    if len(seq_usages) != len(seq_outputs):
                        log_error(
                            f"Expected {len(seq_outputs)} usage entries but got {len(seq_usages)}. Response was: {response}",
                            parameters=self.parameters,
                        )
                    return seq_outputs, seq_usages

                pairs = await asyncio.gather(
                    *(query_one(input_message) for input_message in inputs)
                )
                return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            else:

                async def query_one(
                    input_message: dict,
                ) -> tuple[str, tuple[Optional[int], Optional[int]]]:
                    response = await self.query_client(
                        client,
                        [input_message],
                        max_new_tokens,
                        temperature=temperature,
                        stop_strings=stop_strings,
                    )
                    return self.get_output(response)

                flat = await asyncio.gather(
                    *(
                        query_one(input_message)
                        for input_message in inputs
                        for _ in range(num_return_sequences)
                    )
                )
                outputs = [
                    [
                        flat[i * num_return_sequences + j][0]
                        for j in range(num_return_sequences)
                    ]
                    for i in range(len(inputs))
                ]
                usages = [
                    [
                        flat[i * num_return_sequences + j][1]
                        for j in range(num_return_sequences)
                    ]
                    for i in range(len(inputs))
                ]
                return outputs, usages


class OpenAIAPIModel(OpenAICompatibleAPIBase, APIModel):
    """
    APIModel implementation backed by an OpenAI-compatible client.

    Initializes an ``openai.OpenAI`` client pointed at the given base URL.
    Suitable as a base for any service that exposes an OpenAI-compatible API.
    """

    SUPPORTS_NATIVE_N: bool = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the OpenAI-compatible API model.

        :param model: The model identifier string.
        :type model: str
        :param base_url: The base URL for the OpenAI-compatible API endpoint.
        :type base_url: str
        :param api_key: The API key for authentication. If None, uses environment variables.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: Optional[int]
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    def get_image_input_dict(self, image: str) -> dict:
        """
        Return the OpenAI-format content dict for a base64-encoded image.

        :param image: A base64-encoded JPEG image string.
        :type image: str
        :return: A content dict with ``type`` and ``image_url`` fields.
        :rtype: dict
        """
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }

    async def query_client(
        self,
        client: Any,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> Any:
        """
        Send a message to the OpenAI chat completions endpoint (asynchronously).

        :param messages: A list containing the formatted user message dict.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of sequences to return per prompt.
        :type num_return_sequences: int
        :return: The raw API response object.
        :rtype: Any
        """
        final_stop = list(dict.fromkeys(["[STOP]"] + (stop_strings or [])))
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            stop=final_stop,
            n=num_return_sequences,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        max_tries = 5
        last_error = None
        attempt = 0
        while True:
            try:
                response = await client.chat.completions.create(**kwargs)
                if response is None or not getattr(response, "choices", None):
                    raise ValueError(
                        f"API returned an invalid response (None/missing/empty choices): {response}"
                    )
                return response
            except Exception as e:
                last_error = e
                timeout_retry = _is_timeout(e) and not self.LOCAL_ENDPOINT
                if timeout_retry:
                    max_tries = max(max_tries, TIMEOUT_MAX_TRIES)
                log_warn(
                    f"OpenAI API call failed on attempt {attempt+1}/{max_tries} with error: {e}"
                )
                attempt += 1
                if attempt >= max_tries:
                    break
                backoff_time = _retry_backoff(
                    attempt - 1, self.seconds_to_wait, timeout_retry
                )
                if backoff_time < 0:
                    # Unmetered endpoints (vLLM) carry a negative seconds_to_wait, so
                    # there is no backoff to report — the retry is immediate.
                    log_info("Got connection error with vLLM, trying again.")
                else:
                    log_info(
                        f"Waiting for {backoff_time:.2f} seconds before retrying"
                        f"{' after a timeout' if timeout_retry else ''}..."
                    )
                await asyncio.sleep(backoff_time)
        raise RuntimeError(
            f"OpenAI API call failed after {max_tries} attempts. Last error: {last_error}"
        ) from last_error

    def get_output_texts(
        self, response: Any
    ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
        """
        Extract output text strings and token usage from an OpenAI API response.

        The API reports one ``usage`` for the whole call, so it is emitted on the first
        choice with ``0`` on the rest (``None`` where the count was not reported), and
        summing a record's choices gives the correct total.

        :param response: The raw response object from the OpenAI client.
        :type response: Any
        :return: ``(texts, usages)``, one entry each per choice.
        :rtype: tuple[list[str], list[tuple[Optional[int], Optional[int]]]]
        """
        input_tokens, output_tokens = _extract_usage(
            response,
            input_attr="prompt_tokens",
            output_attr="completion_tokens",
            parameters=self.parameters,
        )
        texts = []
        usages = []
        for choice_index, choice in enumerate(response.choices):
            text = ""
            message = choice.message
            if hasattr(message, "reasoning") and message.reasoning is not None:
                text = "Reasoning: " + message.reasoning
            content = message.content
            if content is not None:
                if text != "":
                    text += "\nResponse: "
                text = text + " " + content
            if text.strip() == "":
                log_warn(f"Received empty output text from model for choice: {choice}")
            texts.append(text.strip())
            if choice_index == 0:
                usages.append((input_tokens, output_tokens))
            else:
                # Already counted on choice 0; None stays None so a genuinely missing
                # count is never mistaken for a zero contribution.
                usages.append(
                    (
                        None if input_tokens is None else 0,
                        None if output_tokens is None else 0,
                    )
                )
        return texts, usages


class OpenAIModel(OpenAIAPIModel):
    """
    Model using the official OpenAI API endpoint.

    Connects directly to OpenAI without a custom base URL.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize an OpenAI model using the default OpenAI endpoint.

        :param model: The OpenAI model identifier (e.g. ``"gpt-4o"``).
        :type model: str
        :param api_key: The OpenAI API key. If None, uses the ``OPENAI_API_KEY`` environment variable.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: Optional[int]
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            base_url=None,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )


class AnthropicModel(APIModel):
    """
    APIModel implementation backed by the Anthropic Messages API.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the Anthropic model.

        :param model: The Anthropic model identifier (e.g. ``"claude-opus-4-6"``).
        :type model: str
        :param api_key: The Anthropic API key. If None, uses the ``ANTHROPIC_API_KEY`` environment variable.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: Optional[int]
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self._async_client_api_key = api_key

    def _make_async_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=self._async_client_api_key)

    def get_image_input_dict(self, image: str) -> dict:
        """
        Return the Anthropic-format content dict for a base64-encoded image.

        :param image: A base64-encoded JPEG image string.
        :type image: str
        :return: A content dict with ``type`` and ``source`` fields.
        :rtype: dict
        """
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image,
            },
        }

    async def query_client(
        self,
        client: Any,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> Any:
        """
        Send a message to the Anthropic messages endpoint (asynchronously).

        :param messages: A list containing the formatted user message dict.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop sequences. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Unused (Anthropic has no native multi-sample API); kept for signature compatibility.
        :type num_return_sequences: int
        :return: The raw API response object.
        :rtype: Any
        """
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_new_tokens)
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs["stop_sequences"] = list(
            dict.fromkeys(["[STOP]"] + (stop_strings or []))
        )
        max_tries = 5
        last_error = None
        attempt = 0
        while True:
            try:
                response = await client.messages.create(**kwargs)
                if response is None or not getattr(response, "content", None):
                    raise ValueError(
                        f"API returned an invalid response (None/missing/empty content): {response}"
                    )
                return response
            except Exception as e:
                last_error = e
                timeout_retry = _is_timeout(e) and not self.LOCAL_ENDPOINT
                if timeout_retry:
                    max_tries = max(max_tries, TIMEOUT_MAX_TRIES)
                log_warn(
                    f"Anthropic API call failed on attempt {attempt+1}/{max_tries} with error: {e}"
                )
                attempt += 1
                if attempt >= max_tries:
                    break
                backoff_time = _retry_backoff(
                    attempt - 1, self.seconds_to_wait, timeout_retry
                )
                log_info(
                    f"Waiting for {backoff_time:.2f} seconds before retrying"
                    f"{' after a timeout' if timeout_retry else ''}..."
                )
                await asyncio.sleep(backoff_time)
        raise RuntimeError(
            f"Anthropic API call failed after {max_tries} attempts. Last error: {last_error}"
        ) from last_error

    def get_output_texts(
        self, response: Any
    ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
        """
        Extract the output text string and token usage from an Anthropic API response.

        Anthropic has no native multi-sample API, so a response always holds exactly one
        sequence and its usage is exact for that sequence.

        :param response: The raw response object from the Anthropic client.
        :type response: Any
        :return: ``(texts, usages)``, each a single-element list.
        :rtype: tuple[list[str], list[tuple[Optional[int], Optional[int]]]]
        """
        text = response.content[0].text
        if text.strip() == "":
            log_warn(f"Received empty output text from model: {response}")
        usage = _extract_usage(
            response,
            input_attr="input_tokens",
            output_attr="output_tokens",
            parameters=self.parameters,
        )
        return [text.strip()], [usage]


class vLLMModel(OpenAIAPIModel):
    """
    Model served via vLLM using an OpenAI-compatible API.

    Uses the OpenAI client pointed at a local or remote vLLM server. The base
    URL is read from ``parameters["vLLM_base_url"]``.
    """

    LOCAL_ENDPOINT: bool = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize a vLLM-served model.

        :param model: The model identifier as registered in the vLLM server.
        :type model: str
        :param api_key: The API key for the vLLM server, if required.
        :type api_key: str or None
        :param base_url: The base url at which the endpoint is accessible, if not default.
        :type base_url: str
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        parameters = load_parameters(parameters)
        if base_url is None:
            base_url = os.environ.get("VLLM_BASE_URL") or parameters["vLLM_base_url"]
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=-1,
            parameters=parameters,
        )


class OpenRouterModel(OpenAIAPIModel):
    """
    Model accessed through the OpenRouter API.

    Routes requests to various model providers (OpenAI, Anthropic, Mistral, etc.)
    via a single OpenAI-compatible endpoint at ``https://openrouter.ai/api/v1``.
    The API key is read from the ``OPENROUTER_API_KEY`` environment variable.
    """

    # OpenRouter does not reliably forward `n` to the underlying provider, so
    # multiple sequences are obtained via separate sequential calls instead.
    SUPPORTS_NATIVE_N: bool = False

    def __init__(
        self,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize an OpenRouter model.

        :param model: The OpenRouter model identifier (e.g. ``"openai/gpt-4o"``).
        :type model: str
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: Optional[int]
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        api_key = os.environ["OPENROUTER_API_KEY"]
        super().__init__(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
