from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode, NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config_schema import Config


@dataclass
class RetrievedSnippet:
    text: str
    score: float
    metadata: Dict[str, Any]


class IndexManager:
    """
    Single Qdrant collection, multiple named vectors:
      - dense_low (low-dim)
      - dense_high (high-dim)
      - bm25_sparse (sparse, BM25-style)

    Query pipeline (toggleable):
      Stage 1: prefetch dense_low and/or bm25_sparse, optional fusion (RRF/DBSF)
      Stage 2: rerank in Qdrant with dense_high (nested prefetch)
      Stage 3: cross-encoder rerank in Python
    """

    DENSE_LOW = "dense_low"
    DENSE_HIGH = "dense_high"
    BM25 = "bm25_sparse"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        self.low_embed = HuggingFaceEmbedding(model_name=cfg.embedding.low_model_name)
        self.high_embed = HuggingFaceEmbedding(model_name=cfg.embedding.high_model_name)

        self.splitter = SentenceSplitter(
            chunk_size=cfg.indexing.chunk_size, chunk_overlap=cfg.indexing.chunk_overlap
        )

        self.cross_rerank = None
        if self.cfg.retrieval.reranker.cross_encoder.enabled:
            self.cross_rerank = SentenceTransformerRerank(
                model=self.cfg.retrieval.reranker.cross_encoder.model,
                top_n=int(self.cfg.retrieval.reranker.cross_encoder.top_n),
            )

        self.client = self._make_qdrant_client(cfg)
        self.collection_name = cfg.qdrant.collection
        self._ensure_collection()

        # Will produce an error if we retrieve dense when we don't store dense
        if (
            self.cfg.retrieval.stage2.dense_high_rerank.enabled
            and not self.cfg.indexing.store_dense_high
        ):
            raise ValueError(
                "stage2.dense_high_rerank enabled but indexing.store_dense_high is false"
            )

        # Will produce an error if we retrieve bm25 when we don't store bm25
        if self.cfg.retrieval.stage1.bm25.enabled and not self.cfg.indexing.store_bm25:
            raise ValueError("stage1.bm25 enabled but indexing.store_bm25 is false")

    def _make_qdrant_client(self, cfg: Config) -> QdrantClient:
        path = cfg.qdrant.path
        if path:
            return QdrantClient(path=cfg.qdrant.path)

        # cloud_inference=True is used for managed cloud inference features;
        # BM25 server-side inference support depends on your Qdrant setup/version.
        return QdrantClient(
            url=cfg.qdrant.url,
            api_key=cfg.qdrant.api_key,
            cloud_inference=getattr(cfg.qdrant, "cloud_inference", False),
        )

    def _ensure_collection(self) -> None:
        # Checking that our collection has the required structure
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            sparse = getattr(info.config.params, "sparse_vectors", None)

            # vectors is either dict(named vectors) or VectorParams(single)
            if not (
                isinstance(vectors, dict)
                and self.DENSE_LOW in vectors
                and self.DENSE_HIGH in vectors
            ):
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' has wrong dense vectors config: {vectors}. "
                    f"Expected named vectors: {self.DENSE_LOW}, {self.DENSE_HIGH}. "
                    "Delete the collection or migrate to a new one."
                )

            if self.cfg.indexing.store_bm25:
                if not (sparse and self.BM25 in sparse):
                    raise RuntimeError(
                        f"Qdrant collection '{self.collection_name}' missing sparse vector '{self.BM25}'. "
                        "Delete the collection or migrate."
                    )

            self._ensure_payload_indexes()
            return

        low_dim = len(self.low_embed.get_text_embedding("dimension probe"))
        high_dim = len(self.high_embed.get_text_embedding("dimension probe"))

        vectors_config: Dict[str, qmodels.VectorParams] = {}
        # Create all spaces we might want to toggle on later
        vectors_config[self.DENSE_LOW] = qmodels.VectorParams(
            size=low_dim, distance=qmodels.Distance.COSINE
        )
        vectors_config[self.DENSE_HIGH] = qmodels.VectorParams(
            size=high_dim, distance=qmodels.Distance.COSINE
        )

        sparse_vectors_config: Optional[Dict[str, qmodels.SparseVectorParams]] = None
        sparse_vectors_config = {
            self.BM25: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        fields = ["tenant_id", "course_id", "document_id", "source_id"]

        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except UnexpectedResponse as e:
                body = (e.content or b"").lower()
                if b"already exist" in body:
                    continue
                raise

    def add_text(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, int]:
        document_id = metadata.get("document_id") or str(uuid.uuid4())

        tenant_id = metadata.get("tenant_id")
        course_id = metadata.get("course_id")
        source_id = metadata.get("source_id")

        if not tenant_id:
            raise ValueError("metadata.tenant_id is required")

        doc = Document(text=text, metadata={**metadata, "document_id": document_id})
        nodes = self.splitter.get_nodes_from_documents([doc])
        texts = [node.get_content() for node in nodes]

        dense_low_vecs = (
            self.low_embed.get_text_embedding_batch(texts)
            if self.cfg.indexing.store_dense_low
            else None
        )
        dense_high_vecs = (
            self.high_embed.get_text_embedding_batch(texts)
            if self.cfg.indexing.store_dense_high
            else None
        )

        points: List[qmodels.PointStruct] = []
        for i, node in enumerate(nodes):
            payload = {
                "text": node.get_content(),
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_index": i,
            }
            if course_id is not None:
                payload["course_id"] = course_id
            if source_id is not None:
                payload["source_id"] = source_id

            vectors: Dict[str, Any] = {}

            if dense_low_vecs is not None:
                vectors[self.DENSE_LOW] = dense_low_vecs[i]

            if dense_high_vecs is not None:
                vectors[self.DENSE_HIGH] = dense_high_vecs[i]

            if self.cfg.indexing.store_bm25:
                vectors[self.BM25] = qmodels.Document(
                    text=node.get_content(),
                    model="Qdrant/bm25",
                    options={
                        "avg_len": float(self.cfg.indexing.bm25_avg_len),
                        "k": float(self.cfg.indexing.bm25_k),
                        "b": float(self.cfg.indexing.bm25_b),
                    },
                )

            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return document_id, len(nodes)

    def _build_stage1_prefetch(
        self, query: str
    ) -> Tuple[Optional[qmodels.Prefetch], Optional[Any], Optional[str]]:
        """
        Returns:
          - stage1_prefetch: a Prefetch (possibly fusion) OR None
          - direct_query: if we skip prefetch and do a direct query, put it here
          - direct_using: vector name for direct query
        """
        prefetch: List[qmodels.Prefetch] = []

        # Dense-low prefetch
        if self.cfg.retrieval.stage1.dense_low.enabled:
            q_low = self.low_embed.get_query_embedding(query)
            prefetch.append(
                qmodels.Prefetch(
                    query=q_low,
                    using=self.DENSE_LOW,
                    limit=int(self.cfg.retrieval.stage1.dense_low.limit),
                )
            )

        # BM25 prefetch
        if self.cfg.retrieval.stage1.bm25.enabled:
            prefetch.append(
                qmodels.Prefetch(
                    query=qmodels.Document(
                        text=query,
                        model="Qdrant/bm25",
                        options={
                            "avg_len": float(self.cfg.indexing.bm25_avg_len),
                            "k": float(self.cfg.indexing.bm25_k),
                            "b": float(self.cfg.indexing.bm25_b),
                        },
                    ),
                    using=self.BM25,
                    limit=int(self.cfg.retrieval.stage1.bm25.limit),
                )
            )

        if not prefetch:
            return None, None, None

        # If only one retriever enabled (and fusion disabled), we can just do direct query
        if len(prefetch) == 1 and not self.cfg.retrieval.stage1.fusion.enabled:
            only = prefetch[0]
            return None, only.query, only.using

        # Otherwise fuse stage1 results
        fusion_method = str(self.cfg.retrieval.stage1.fusion.method).lower()
        fusion = qmodels.Fusion.RRF if fusion_method == "rrf" else qmodels.Fusion.DBSF

        stage1 = qmodels.Prefetch(
            prefetch=prefetch,
            query=qmodels.FusionQuery(fusion=fusion),
            limit=int(self.cfg.retrieval.stage1.fusion.limit),
        )
        return stage1, None, None

    def retrieve(
        self,
        query: str,
        threshold: Optional[float] = None,
        tenant_id: Optional[str] = None,
        course_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedSnippet]:
        must = []
        if tenant_id:
            must.append(
                qmodels.FieldCondition(
                    key="tenant_id", match=qmodels.MatchValue(value=tenant_id)
                )
            )
        if course_id:
            must.append(
                qmodels.FieldCondition(
                    key="course_id", match=qmodels.MatchValue(value=course_id)
                )
            )

        query_filter = qmodels.Filter(must=must) if must else None

        effective_top_k = (
            int(top_k) if top_k is not None else int(self.cfg.retrieval.top_k)
        )
        stage2_limit = int(self.cfg.retrieval.stage2.dense_high_rerank.limit)
        stage2_limit = max(stage2_limit, effective_top_k)

        stage1_prefetch, direct_query, direct_using = self._build_stage1_prefetch(query)

        score_threshold = None
        if threshold is not None:
            score_threshold = float(threshold)
        elif self.cfg.retrieval.post.score_threshold.enabled:
            score_threshold = float(self.cfg.retrieval.post.score_threshold.value)

        # Stage2: dense_high rerank in Qdrant
        if self.cfg.retrieval.stage2.dense_high_rerank.enabled:
            q_high = self.high_embed.get_query_embedding(query)

            # If stage1_prefetch exists, rerank its candidates.
            # Else rerank directly from the chosen direct_query/direct_using (or fallback dense_high only).
            if stage1_prefetch is not None:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[stage1_prefetch],
                    query=q_high,
                    using=self.DENSE_HIGH,
                    limit=stage2_limit,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
            elif direct_query is not None and direct_using is not None:
                # Multistage: prefetch single retriever, then rerank with dense_high
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        qmodels.Prefetch(
                            query=direct_query,
                            using=direct_using,
                            limit=int(self.cfg.retrieval.stage1.dense_low.limit)
                            if direct_using == self.DENSE_LOW
                            else int(self.cfg.retrieval.stage1.bm25.limit),
                        )
                    ],
                    query=q_high,
                    using=self.DENSE_HIGH,
                    limit=stage2_limit,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
            else:
                # Fallback: dense_high only
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=q_high,
                    using=self.DENSE_HIGH,
                    limit=stage2_limit,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )

            points = response.points

        else:
            # No stage2 rerank: either fused stage1, or direct stage1, or dense_high only
            if stage1_prefetch is not None:
                fusion_method = str(self.cfg.retrieval.stage1.fusion.method).lower()
                fusion = (
                    qmodels.Fusion.RRF
                    if fusion_method == "rrf"
                    else qmodels.Fusion.DBSF
                )
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=stage1_prefetch.prefetch,
                    query=qmodels.FusionQuery(fusion=fusion),
                    limit=effective_top_k,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
                points = response.points
            elif direct_query is not None and direct_using is not None:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=direct_query,
                    using=direct_using,
                    limit=effective_top_k,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
                points = response.points
            else:
                q_high = self.high_embed.get_query_embedding(query)
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=q_high,
                    using=self.DENSE_HIGH,
                    limit=effective_top_k,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
                points = response.points

        # Convert Qdrant points -> NodeWithScore for optional reranker
        nodes: List[NodeWithScore] = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            node = TextNode(text=text, metadata=meta)
            nodes.append(NodeWithScore(node=node, score=float(point.score or 0.0)))

        # Optional cross-encoder reranker (Python-side)
        if self.cross_rerank and nodes:
            nodes = self.cross_rerank.postprocess_nodes(nodes, query_str=query)

        nodes = nodes[:effective_top_k]

        return [
            RetrievedSnippet(
                text=node.node.get_content(),
                score=float(node.score or 0.0),
                metadata=dict(node.node.metadata or {}),
            )
            for node in nodes
        ]
