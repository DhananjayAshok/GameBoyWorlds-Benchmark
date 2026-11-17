from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from gameboy_worlds.emulation.parser import StateParser

from utils.log_handling import log_error, log_info
from utils.parameter_handling import load_parameters
from utils.vlm import convert_numpy_greyscale_to_pillow

_EmbeddingInput = Union[str, np.ndarray, Image.Image]


class RandomPatchProjection:
    """Game Boy specific: assumes a 160x144 screen split into 90 cells of 16x16."""

    cell_reduction_dimension = 8

    def __init__(self, parameters: dict = None):
        parameters = load_parameters(parameters)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        start = 16 * 16
        end = self.cell_reduction_dimension
        my_local_rng = torch.Generator(device=device)
        my_local_rng.manual_seed(parameters["random_seed"])
        step1 = nn.Linear(start, end, bias=False, dtype=torch.bfloat16, device=device)
        nn.init.kaiming_normal_(step1.weight, generator=my_local_rng)
        self.random_projection = nn.Sequential(step1)

    def _embed_single(self, item: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        item = np.array(item)
        grid_cells = StateParser.capture_grid_cells(item, y_offset=0)
        cell_embeddings = []
        for key in sorted(grid_cells.keys()):
            cell_image_resized = np.resize(grid_cells[key], (16, 16))
            cell_image_tensor = torch.tensor(
                cell_image_resized.flatten(),
                dtype=torch.bfloat16,
                device=self.random_projection[0].weight.device,
            )
            with torch.no_grad():
                cell_embedding = self.random_projection(cell_image_tensor)
                cell_embedding = cell_embedding / (cell_embedding.norm() + 1e-6)
            cell_embeddings.append(cell_embedding)
        return torch.cat(cell_embeddings, dim=0)

    def project(self, items: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        return torch.stack([self._embed_single(item) for item in items], dim=0)


class HuggingFaceEmbeddingEngine(ABC):
    MODEL_REGISTRY: Dict[str, Tuple[AutoModel, AutoProcessor, str]] = {}
    """ Model registry to cache loaded models. Keyed by model name. Value is a tuple of (model, processor, model_kind). """

    @staticmethod
    @abstractmethod
    def _do_start(model_kind: str, model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        raise NotImplementedError()

    @staticmethod
    def start(
        engine_class: Type["HuggingFaceEmbeddingEngine"],
        model_kind: str,
        model_name: str,
    ):
        if model_name in HuggingFaceEmbeddingEngine.MODEL_REGISTRY:
            _model, _processor, loaded_model_kind = (
                HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
            )
            if loaded_model_kind != model_kind:
                log_error(
                    f"Model '{model_name}' is already loaded with model_kind "
                    f"'{loaded_model_kind}', but tried to load with different model_kind "
                    f"'{model_kind}'"
                )
            return
        log_info(f"Loading HuggingFace embedding model: {model_name} | {model_kind}")
        model, processor = engine_class._do_start(model_kind, model_name)
        HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name] = (
            model,
            processor,
            model_kind,
        )

    @staticmethod
    def is_loaded(model_name: str) -> bool:
        return model_name in HuggingFaceEmbeddingEngine.MODEL_REGISTRY

    @staticmethod
    @abstractmethod
    def _do_embed(
        model_kind: str, model_name: str, items: List[_EmbeddingInput]
    ) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def embed(
        engine_class: Type["HuggingFaceEmbeddingEngine"],
        model_kind: str,
        model_name: str,
        items: List[_EmbeddingInput],
    ) -> torch.Tensor:
        if not HuggingFaceEmbeddingEngine.is_loaded(model_name=model_name):
            HuggingFaceEmbeddingEngine.start(
                engine_class=engine_class, model_kind=model_kind, model_name=model_name
            )
        return engine_class._do_embed(
            model_kind=model_kind, model_name=model_name, items=items
        )


class HuggingFaceTextEmbeddingEngine(HuggingFaceEmbeddingEngine):

    @staticmethod
    def _do_start(model_kind: str, model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        if model_kind == "qwen3":
            model = AutoModel.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto"
            )
            processor = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            return model, processor
        elif model_kind == "jina":
            # jina's remote code does its own device placement; pin to GPU, not device_map.
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda")
            return model, None
        else:
            log_error(f"Unsupported text embedding model kind: {model_kind}")

    @staticmethod
    def _do_embed(
        model_kind: str, model_name: str, items: List[_EmbeddingInput]
    ) -> torch.Tensor:
        model, processor, _ = HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
        if model_kind == "qwen3":
            max_length = load_parameters()["text_embedding_model_max_length"]
            inputs = processor(
                items,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, -1]
            return embeddings.detach().cpu()
        elif model_kind == "jina":
            passage_embeddings = model.encode_text(texts=items, task="retrieval")
            return torch.stack(passage_embeddings).cpu()
        else:
            log_error(f"Text embedding model kind {model_kind} not implemented.")


class HuggingFaceImageEmbeddingEngine(HuggingFaceEmbeddingEngine):
    @staticmethod
    def _do_start(model_kind: str, model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        if model_kind == "jina":
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda")
            return model, None
        elif model_kind == "random_patch":
            return RandomPatchProjection(), None
        else:
            log_error(f"Unsupported image embedding model kind: {model_kind}")

    @staticmethod
    def _do_embed(
        model_kind: str, model_name: str, items: List[_EmbeddingInput]
    ) -> torch.Tensor:
        model, _processor, _ = HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
        if model_kind == "random_patch":
            return model.project(items)
        images = []
        for item in items:
            if isinstance(item, np.ndarray):
                images.append(convert_numpy_greyscale_to_pillow(item))
            elif isinstance(item, Image.Image):
                images.append(item)
            else:
                log_error(
                    f"Unsupported input type for image embedding: {type(item)}"
                )
        if model_kind == "jina":
            embeddings = model.encode_image(images=images, task="retrieval")
            return torch.stack(embeddings).cpu()
        else:
            log_error(f"Image embedding model kind {model_kind} not implemented.")


class EmbeddingModel(ABC):
    def __init__(
        self,
        model_name: str,
        model_kind: str,
        engine: Type[HuggingFaceEmbeddingEngine],
    ):
        self._model_name = model_name
        self._model_kind = model_kind
        self._ENGINE = engine
        if not issubclass(engine, HuggingFaceEmbeddingEngine):
            log_error("EmbeddingModel only supports HuggingFaceEmbeddingEngine currently.")
        self._ENGINE.start(
            self._ENGINE, model_kind=self._model_kind, model_name=self._model_name
        )

    def embed(self, items: List[_EmbeddingInput]) -> torch.Tensor:
        return self._ENGINE.embed(
            engine_class=self._ENGINE,
            model_kind=self._model_kind,
            model_name=self._model_name,
            items=items,
        )

    def compare(
        self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(
            embeddings_a / embeddings_a.norm(dim=1, keepdim=True),
            (embeddings_b / embeddings_b.norm(dim=1, keepdim=True)).T,
        )

    def embed_compare(
        self, item: _EmbeddingInput, existing_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embed([item])
        similarity_scores = self.compare(embedding, existing_index)
        return embedding, similarity_scores


class TextEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name: str = None, model_kind: str = None, parameters: dict = None):
        if model_name is None:
            parameters = load_parameters(parameters)
            model_name = parameters["text_embedding_model"]
            model_kind = parameters["text_embedding_model_kind"]
        super().__init__(
            model_name=model_name,
            model_kind=model_kind,
            engine=HuggingFaceTextEmbeddingEngine,
        )


class ImageEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = None, model_kind: str = None, parameters: dict = None):
        if model_name is None:
            parameters = load_parameters(parameters)
            model_name = parameters["image_embedding_model"]
            model_kind = parameters["image_embedding_model_kind"]
        super().__init__(
            model_name=model_name,
            model_kind=model_kind,
            engine=HuggingFaceImageEmbeddingEngine,
        )


class Index:
    def __init__(self, modality: str, parameters: dict = None):
        if modality not in ["text", "image"]:
            log_error(f"Index modality must be 'text' or 'image', got {modality}")
        self.modality = modality
        if modality == "text":
            self._embedding_model = TextEmbeddingModel(parameters=parameters)
        else:
            self._embedding_model = ImageEmbeddingModel(parameters=parameters)
        self.index = None
        """ The tensor index of embeddings. Shape: (num_entries, embed_size) """

    def embed(self, items: List[_EmbeddingInput]) -> torch.Tensor:
        if not isinstance(items, list):
            items = [items]
        return self._embedding_model.embed(items)

    def add_embedding_to_index(self, embedding: torch.Tensor):
        if self.index is None:
            self.index = embedding
        else:
            self.index = torch.cat([self.index, embedding], dim=0)

    def add_to_index(self, items: List[_EmbeddingInput]):
        self.add_embedding_to_index(self.embed(items))

    def add_compare(self, item: _EmbeddingInput) -> torch.Tensor:
        """Scores are against the index before this item was added; None if it was empty."""
        if self.index is None:
            self.add_to_index([item])
            return None
        embedding, similarity_scores = self._embedding_model.embed_compare(
            item, self.index
        )
        self.add_embedding_to_index(embedding)
        return similarity_scores


