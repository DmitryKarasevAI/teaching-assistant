from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import Document

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

from qdrant_client import QdrantClient, models as qdrant_models
from llama_index.vector_stores.qdrant import QdrantVectorStore

from ..config_schema import Config


@dataclass
class RetrievedSnippet:
    text: str
    score: float
    metadata: Dict[str, Any]


class IndexManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.embedding.model_name)
        Settings.llm = HuggingFaceLLM(
            model_name=cfg.llm.model_name,
            tokenizer_name=cfg.llm.tokenizer,
        )

        self.splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)

        self.client = self._make_qdrant_client(cfg)
        self.collection_name = cfg.qdrant.collection

        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            prefer_grpc=cfg.qdrant.prefer_grpc,
        )

        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

    @staticmethod
    def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value if value else None

    def _make_qdrant_client(self, cfg: Config) -> QdrantClient:
        path = self._normalize_optional_str(cfg.qdrant.path)
        if path:
            return QdrantClient(path=path)

        return QdrantClient(
            url=cfg.qdrant.url,
            api_key=cfg.qdrant.api_key,
        )

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            return

        dim = len(Settings.embed_model.get_text_embedding("dimension probe"))

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text-dense": qdrant_models.VectorParams(
                    size=dim,
                    distance=qdrant_models.Distance.COSINE,
                )
            },
        )

    def add_text(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, int]:
        document_id = metadata.get("document_id") or str(uuid.uuid4())
        metadata = {**metadata, "document_id": document_id}

        doc = Document(text=text, metadata=metadata)

        nodes = self.splitter.get_nodes_from_documents([doc])

        self.index.insert_nodes(nodes)

        return document_id, len(nodes)

    def retrieve(
        self, query: str, threshold: Optional[float] = None
    ) -> List[RetrievedSnippet]:
        retriever = self.index.as_retriever(similarity_top_k=self.cfg.app.top_k)
        results = retriever.retrieve(query)

        if threshold is None:
            threshold = self.cfg.app.threshold

        if threshold is not None:
            results = SimilarityPostprocessor(
                similarity_cutoff=float(threshold)
            ).postprocess_nodes(results)

        snippets: List[RetrievedSnippet] = []
        for r in results:
            snippets.append(
                RetrievedSnippet(
                    text=r.node.get_content(),
                    score=float(r.score or 0.0),
                    metadata=dict(r.node.metadata or {}),
                )
            )
        return snippets
