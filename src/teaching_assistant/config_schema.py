from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en"


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer: str = "Qwen/Qwen3-0.6B"


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    path: Optional[str] = None
    collection: str = "teaching-assistant"
    prefer_grpc: bool = False


@dataclass
class AppConfig:
    persist_dir: str = "data/index"
    raw_docs_dir: str = "data/raw_docs"
    top_k: int = 10
    threshold: float = 0.85
    num_questions: int = 8


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
