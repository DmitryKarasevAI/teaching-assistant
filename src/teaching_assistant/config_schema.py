from dataclasses import dataclass, field
from typing import Optional


# ---------- Embeddings / LLM ----------


@dataclass
class EmbeddingConfig:
    low_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    high_model_name: str = "BAAI/bge-large-en-v1.5"


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer: str = "Qwen/Qwen3-0.6B"


# ---------- Qdrant ----------


@dataclass
class QdrantConfig:
    url: str = "http://qdrant:6333"
    api_key: Optional[str] = None
    path: Optional[str] = None
    collection: str = "teaching-assistant"
    prefer_grpc: bool = False
    cloud_inference: bool = False  # for Qdrant Cloud inference features


# ---------- Indexing controls ----------


@dataclass
class IndexingConfig:
    store_dense_low: bool = True
    store_dense_high: bool = True
    store_bm25: bool = True

    # BM25 options used by Qdrant inference Document(...) (avg_len is important)
    bm25_avg_len: float = 120.0
    bm25_k: float = 1.2
    bm25_b: float = 0.75

    chunk_size: int = 1024
    chunk_overlap: int = 256


# ---------- Retrieval pipeline ----------


@dataclass
class DenseLowStageConfig:
    enabled: bool = True
    limit: int = 200


@dataclass
class BM25StageConfig:
    enabled: bool = True
    limit: int = 200


@dataclass
class FusionConfig:
    enabled: bool = True
    method: str = "rrf"  # "rrf" or "dbsf"
    limit: int = 200


@dataclass
class Stage1Config:
    dense_low: DenseLowStageConfig = field(default_factory=DenseLowStageConfig)
    bm25: BM25StageConfig = field(default_factory=BM25StageConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class DenseHighRerankConfig:
    enabled: bool = True
    limit: int = 50


@dataclass
class Stage2Config:
    dense_high_rerank: DenseHighRerankConfig = field(
        default_factory=DenseHighRerankConfig
    )


@dataclass
class ScoreThresholdConfig:
    enabled: bool = False
    value: float = 0.25


@dataclass
class PostConfig:
    score_threshold: ScoreThresholdConfig = field(default_factory=ScoreThresholdConfig)


@dataclass
class CrossEncoderConfig:
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    top_n: int = 20


@dataclass
class RerankerConfig:
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)


@dataclass
class RetrievalConfig:
    top_k: int = 20
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    post: PostConfig = field(default_factory=PostConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


# ---------- App-level stuff you still use ----------


@dataclass
class AppConfig:
    persist_dir: str = "data/index"
    raw_docs_dir: str = "data/raw_docs"
    threshold: float = 0.4


# ---------- Root ----------


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
