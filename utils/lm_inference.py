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


MARKUP_CHARS = "*`"


def strip_markup(value: str, chars: str = MARKUP_CHARS) -> str:
    while True:
        stripped = value.strip().strip(chars).strip()
        if len(stripped) > 1 and stripped[0] == "<" and stripped[-1] == ">":
            stripped = stripped[1:-1]
        if stripped == value:
            return value
        value = stripped


def unmark_line(line: str) -> str:
    return line.replace("**", "").replace("__", "").replace("`", "")


def parse_key_value(text: str, key: str) -> Optional[str]:
    key_lower = key.lower()
    marker = f"{key_lower}:"

    def clean(value: str) -> Optional[str]:
        value = strip_markup(value)
        return value or None

    text_lower = text.lower()
    if text_lower.count("response:") == 1:
        idx = text_lower.index("response:") + len("response:")
        text = text[idx:].strip()

    lines = [unmark_line(line) for line in text.splitlines()]
    for line in lines:
        start = line.strip().lstrip("#-*• ").strip()
        if start.lower().startswith(marker):
            return clean(start[len(marker):])
    for line in lines:
        idx = line.lower().find(marker)
        if idx != -1:
            return clean(line[idx + len(marker) :])

    text = "\n".join(lines)
    text_lower = text.lower()
    if text_lower.count(key_lower) == 1:
        idx = text_lower.index(key_lower)
        rest_of_line = text[idx + len(key) :].splitlines()
        return clean(rest_of_line[0]) if rest_of_line else None

    return None


def clean_value(value: Optional[str]) -> Optional[str]:
    value = strip_markup(value or "", MARKUP_CHARS + "\"'")
    if not value or value.lower() in ("none", "n/a", "na", "unknown"):
        return None
    return value


def parse_yes_no(text: str, key: str) -> Optional[bool]:
    raw = parse_key_value(text, key)
    if raw is None:
        return None
    return raw.strip().lower().startswith("yes")


MIN_QUERIES_PER_MINUTE = 1

# A timeout is a provider stall, not rate limiting, so it gets a much longer backoff; local vLLM is excluded.
TIMEOUT_MAX_TRIES = 20
TIMEOUT_BACKOFF_FLOOR = 60.0
TIMEOUT_BACKOFF_CAP = 600.0


def _is_timeout(error: Exception) -> bool:
    return isinstance(error, (APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError))


def _retry_backoff(attempt: int, seconds_to_wait: float, timeout_retry: bool) -> float:
    # Jitter spreads out concurrent calls that failed together.
    if timeout_retry:
        backoff = TIMEOUT_BACKOFF_FLOOR * (2**attempt) * random.uniform(1.0, 1.5)
        return min(backoff, TIMEOUT_BACKOFF_CAP)
    return seconds_to_wait * (2**attempt) * random.uniform(1.0, 1.5)


# Queries per minute.
_RATE_LIMITS: dict[str, int] = {
    "gpt-4o-mini": 60,
    "gpt-4o": 60,
    "gpt-4": 60,
    "gpt-5": 60,
    "claude-opus-4.7": 60,
    "claude-sonnet-4-6": 60,
    "claude-haiku-4-5-20251001": 60,
    "google/gemini-3.1-pro-preview": 60,
    "google/gemini-3.6-flash": 200,
    "google/gemini-4-argon": 120,
    "qwen/qwen3-vl-235b-a22b-instruct": 60,
    "google/gemma-4-31b-it": 1000,
}


_WARNED_MISSING_RATE_LIMIT: set[str] = set()


def get_max_queries_per_minute(model: str, parameters: dict[str, Any]) -> int:
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
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def sum_meta(*metas: dict[str, Any]) -> dict[str, Optional[int]]:
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
    return {"input_tokens": 0, "output_tokens": 0}


def _collapse_meta(meta: dict[str, list[Optional[int]]]) -> dict[str, Optional[int]]:
    return {key: value[0] for key, value in meta.items()}


def _extract_usage(
    response: Any,
    *,
    input_attr: str,
    output_attr: str,
    parameters: dict[str, Any] = None,
) -> tuple[Optional[int], Optional[int]]:
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
        time_to_wait = self.seconds_to_wait - (perf_counter() - self.last_query_time)
        if time_to_wait > 0:
            sleep(time_to_wait)
        self.last_query_time = perf_counter()


class OpenAICompatibleAPIBase(RateLimitedAPIBase):
    # The client is built fresh inside each asyncio.run(): its connection pool must not cross event loops.

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
    """Abstract base class for all LM inference backends."""

    BATCH_SIZE_PARAMETER: str = "max_batch_size_api"

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
        pass

    def _build_meta(
        self, *, usages: list[list[tuple[Optional[int], Optional[int]]]]
    ) -> dict[str, list[Optional[int]]]:
        # meta is per record: counts are summed over a record's sequences, and None (unreported) propagates.
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
        stop_idx = output_text.lower().find("[stop]")
        if stop_idx != -1:
            output_text = output_text[:stop_idx]
        return output_text.strip()

    def _standardize_format(
        self,
        texts: Union[str, list[str]],
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
    ) -> tuple[list[str], list[list[Image.Image]], bool]:
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
        texts, images, passed_in_str = self._standardize_format(texts, images)
        parameters = (
            self.parameters if hasattr(self, "parameters") else load_parameters()
        )
        if batch_size is None:
            batch_size = parameters[self.BATCH_SIZE_PARAMETER]
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
        # meta here is a scalar total over both passes, not per-record.
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
        )
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
    SUPPORTS_NATIVE_N: bool = False

    # True for endpoints we run ourselves (vLLM), where a timeout means a dead server, not a provider stall.
    LOCAL_ENDPOINT: bool = False

    def __init__(
        self,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    def get_encoded_images(self, images: list[Image.Image]) -> list[str]:
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
        pass

    @abstractmethod
    def _make_async_client(self) -> Any:
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
        pass

    @abstractmethod
    def get_output_texts(
        self, response: Any
    ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
        # With native n, usage goes on the first entry and siblings carry 0 (already counted); None means unreported.
        pass

    def get_outputs(
        self, response: Any
    ) -> tuple[list[str], list[tuple[Optional[int], Optional[int]]]]:
        texts, usages = self.get_output_texts(response)
        return [self.get_output_final(t) for t in texts], usages

    def get_output(
        self, response: Any
    ) -> tuple[str, tuple[Optional[int], Optional[int]]]:
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
        # One wait() paces the whole batch; its requests then fire concurrently.
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
    SUPPORTS_NATIVE_N: bool = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    def get_image_input_dict(self, image: str) -> dict:
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
                    # Unmetered endpoints (vLLM) carry a negative seconds_to_wait, so the retry is immediate.
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
                # Already counted on choice 0; None stays None so a missing count is not read as zero.
                usages.append(
                    (
                        None if input_tokens is None else 0,
                        None if output_tokens is None else 0,
                    )
                )
        return texts, usages


class OpenAIModel(OpenAIAPIModel):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=None,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )


class AnthropicModel(APIModel):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self._async_client_api_key = api_key

    def _make_async_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=self._async_client_api_key)

    def get_image_input_dict(self, image: str) -> dict:
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
    LOCAL_ENDPOINT: bool = True
    BATCH_SIZE_PARAMETER: str = "max_batch_size_vllm"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
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
    # OpenRouter does not reliably forward `n` to the provider, so each sequence is a separate call.
    SUPPORTS_NATIVE_N: bool = False

    def __init__(
        self,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        api_key = os.environ["OPENROUTER_API_KEY"]
        super().__init__(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )


class GatewayModel(OpenAIAPIModel):
    SUPPORTS_NATIVE_N: bool = False

    def __init__(
        self,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=os.environ["MODEL_GATEWAY_URL"],
            api_key="none",
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
