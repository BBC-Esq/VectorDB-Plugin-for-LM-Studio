# draft of langchain's wrapper around tiledb and tiledb-vector-search implementing int8, uint8, and two other distance metrics

from __future__ import annotations
import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance

INDEX_METRICS = frozenset(["euclidean", "l2", "cosine", "sum_of_squares"])
DEFAULT_METRIC = "euclidean"
DOCUMENTS_ARRAY_NAME = "documents"
VECTOR_INDEX_NAME = "vectors"
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_FLOAT_32 = np.finfo(np.dtype("float32")).max
MAX_FLOAT = sys.float_info.max

def dependable_tiledb_import() -> Any:
    return (
        guard_import("tiledb.vector_search"),
        guard_import("tiledb"),
    )

def get_vector_index_uri_from_group(group: Any) -> str:
    return group[VECTOR_INDEX_NAME].uri

def get_documents_array_uri_from_group(group: Any) -> str:
    return group[DOCUMENTS_ARRAY_NAME].uri

def get_vector_index_uri(uri: str) -> str:
    return f"{uri}/{VECTOR_INDEX_NAME}"

def get_documents_array_uri(uri: str) -> str:
    return f"{uri}/{DOCUMENTS_ARRAY_NAME}"

class TileDB(VectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        index_uri: str,
        metric: str,
        *,
        vector_index_uri: str = "",
        docs_array_uri: str = "",
        config: Optional[Mapping[str, Any]] = None,
        timestamp: Any = None,
        allow_dangerous_deserialization: bool = False,
        **kwargs: Any,
    ):
        if not allow_dangerous_deserialization:
            raise ValueError(
                "TileDB relies on pickle for serialization and deserialization. "
                "This can be dangerous if the data is intercepted and/or modified "
                "by malicious actors prior to being de-serialized. "
                "If you are sure that the data is safe from modification, you can "
                " set allow_dangerous_deserialization=True to proceed. "
                "Loading of compromised data using pickle can result in execution of "
                "arbitrary code on your machine."
            )
        self.embedding = embedding
        self.embedding_function = embedding.embed_query
        self.index_uri = index_uri
        self.metric = metric
        self.config = config
        tiledb_vs, tiledb = (
            guard_import("tiledb.vector_search"),
            guard_import("tiledb"),
        )
        with tiledb.scope_ctx(ctx_or_config=config):
            index_group = tiledb.Group(self.index_uri, "r")
            self.vector_index_uri = (
                vector_index_uri
                if vector_index_uri != ""
                else get_vector_index_uri_from_group(index_group)
            )
            self.docs_array_uri = (
                docs_array_uri
                if docs_array_uri != ""
                else get_documents_array_uri_from_group(index_group)
            )
            index_group.close()
            group = tiledb.Group(self.vector_index_uri, "r")
            self.index_type = group.meta.get("index_type")
            group.close()
            self.timestamp = timestamp
            if self.index_type == "FLAT":
                self.vector_index = tiledb_vs.flat_index.FlatIndex(
                    uri=self.vector_index_uri,
                    config=self.config,
                    timestamp=self.timestamp,
                    **kwargs,
                )
            elif self.index_type == "IVF_FLAT":
                self.vector_index = tiledb_vs.ivf_flat_index.IVFFlatIndex(
                    uri=self.vector_index_uri,
                    config=self.config,
                    timestamp=self.timestamp,
                    **kwargs,
                )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def process_index_results(
        self,
        ids: List[int],
        scores: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = MAX_FLOAT,
    ) -> List[Tuple[Document, float]]:
        tiledb = guard_import("tiledb")
        docs = []
        docs_array = tiledb.open(
            self.docs_array_uri, "r", timestamp=self.timestamp, config=self.config
        )
        for idx, score in zip(ids, scores):
            if idx == 0 and score == 0:
                continue
            if idx == MAX_UINT64 and score == MAX_FLOAT_32:
                continue
            doc = docs_array[idx]
            if doc is None or len(doc["text"]) == 0:
                raise ValueError(f"Could not find document for id {idx}, got {doc}")
            pickled_metadata = doc.get("metadata")
            result_doc = Document(page_content=str(doc["text"][0]))
            if pickled_metadata is not None:
                metadata = pickle.loads(
                    np.array(pickled_metadata.tolist()).astype(np.uint8).tobytes()
                )
                result_doc.metadata = metadata
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(
                    result_doc.metadata.get(key) in value
                    for key, value in filter.items()
                ):
                    docs.append((result_doc, score))
            else:
                docs.append((result_doc, score))
        docs_array.close()
        docs = [(doc, score) for doc, score in docs if score <= score_threshold]
        return docs[:k]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        if "score_threshold" in kwargs:
            score_threshold = kwargs.pop("score_threshold")
        else:
            score_threshold = MAX_FLOAT
        embedding_array = np.array(embedding, copy=False)
        d, i = self.vector_index.query(
            np.array([embedding_array]),
            k=k if filter is None else fetch_k,
            **kwargs,
        )
        return self.process_index_results(
            ids=i[0], scores=d[0], filter=filter, k=k, score_threshold=score_threshold
        )

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        if "score_threshold" in kwargs:
            score_threshold = kwargs.pop("score_threshold")
        else:
            score_threshold = MAX_FLOAT
        embedding_array = np.array(embedding, copy=False)
        scores, indices = self.vector_index.query(
            np.array([embedding_array]),
            k=fetch_k if filter is None else fetch_k * 2,
            **kwargs,
        )
        results = self.process_index_results(
            ids=indices[0],
            scores=scores[0],
            filter=filter,
            k=fetch_k if filter is None else fetch_k * 2,
            score_threshold=score_threshold,
        )
        embeddings = [
            self.embedding.embed_documents([doc.page_content])[0] for doc, _ in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding_array]),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        docs_and_scores = []
        for i in mmr_selected:
            docs_and_scores.append(results[i])
        return docs_and_scores

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    @classmethod
    def create(
        cls,
        index_uri: str,
        index_type: str,
        dimensions: int,
        vector_type: np.dtype,
        *,
        metadatas: bool = True,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        tiledb_vs, tiledb = (
            guard_import("tiledb.vector_search"),
            guard_import("tiledb"),
        )
        with tiledb.scope_ctx(ctx_or_config=config):
            try:
                tiledb.group_create(index_uri)
            except tiledb.TileDBError as err:
                raise err
            group = tiledb.Group(index_uri, "w")
            vector_index_uri = get_vector_index_uri(group.uri)
            docs_uri = get_documents_array_uri(group.uri)
            if index_type == "FLAT":
                tiledb_vs.flat_index.create(
                    uri=vector_index_uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    config=config,
                )
            elif index_type == "IVF_FLAT":
                tiledb_vs.ivf_flat_index.create(
                    uri=vector_index_uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    config=config,
                )
            group.add(vector_index_uri, name=VECTOR_INDEX_NAME)
            dim = tiledb.Dim(
                name="id",
                domain=(0, MAX_UINT64 - 1),
                dtype=np.dtype(np.uint64),
            )
            dom = tiledb.Domain(dim)
            text_attr = tiledb.Attr(name="text", dtype=np.dtype("U1"), var=True)
            attrs = [text_attr]
            if metadatas:
                metadata_attr = tiledb.Attr(name="metadata", dtype=np.uint8, var=True)
                attrs.append(metadata_attr)
            schema = tiledb.ArraySchema(
                domain=dom,
                sparse=True,
                allows_duplicates=False,
                attrs=attrs,
            )
            tiledb.Array.create(docs_uri, schema)
            group.add(docs_uri, name=DOCUMENTS_ARRAY_NAME)
            group.close()

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        index_uri: str,
        *,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> TileDB:
        if metric not in INDEX_METRICS:
            raise ValueError(
                f"Unsupported distance metric: {metric}. Expected one of {list(INDEX_METRICS)}"
            )
        tiledb_vs, tiledb = (
            guard_import("tiledb.vector_search"),
            guard_import("tiledb"),
        )
        input_vectors = np.array(embeddings)
        cls.create(
            index_uri=index_uri,
            index_type=index_type,
            dimensions=input_vectors.shape[1],
            vector_type=input_vectors.dtype,
            metadatas=metadatas is not None,
            config=config,
        )
        with tiledb.scope_ctx(ctx_or_config=config):
            if not embeddings:
                raise ValueError("embeddings must be provided to build a TileDB index")
            vector_index_uri = get_vector_index_uri(index_uri)
            docs_uri = get_documents_array_uri(index_uri)
            if ids is None:
                ids = [str(random.randint(0, MAX_UINT64 - 1)) for _ in texts]
            external_ids = np.array(ids).astype(np.uint64)
            tiledb_vs.ingestion.ingest(
                index_type=index_type,
                index_uri=vector_index_uri,
                input_vectors=input_vectors,
                external_ids=external_ids,
                index_timestamp=index_timestamp if index_timestamp != 0 else None,
                config=config,
                **kwargs,
            )
            with tiledb.open(docs_uri, "w") as A:
                if external_ids is None:
                    external_ids = np.zeros(len(texts), dtype=np.uint64)
                    for i in range(len(texts)):
                        external_ids[i] = i
                data = {}
                data["text"] = np.array(texts)
                if metadatas is not None:
                    metadata_attr = np.empty([len(metadatas)], dtype=object)
                    i = 0
                    for metadata in metadatas:
                        metadata_attr[i] = np.frombuffer(
                            pickle.dumps(metadata), dtype=np.uint8
                        )
                        i += 1
                    data["metadata"] = metadata_attr
                A[external_ids] = data
        return cls(
            embedding=embedding,
            index_uri=index_uri,
            metric=metric,
            config=config,
            **kwargs,
        )

    def delete(
        self, ids: Optional[List[str]] = None, timestamp: int = 0, **kwargs: Any
    ) -> Optional[bool]:
        external_ids = np.array(ids).astype(np.uint64)
        self.vector_index.delete_batch(
            external_ids=external_ids, timestamp=timestamp if timestamp != 0 else None
        )
        return True

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        timestamp: int = 0,
        **kwargs: Any,
    ) -> List[str]:
        tiledb = guard_import("tiledb")
        embeddings = self.embedding.embed_documents(list(texts))
        if ids is None:
            ids = [str(random.randint(0, MAX_UINT64 - 1)) for _ in texts]
        external_ids = np.array(ids).astype(np.uint64)
        vectors = np.empty((len(embeddings)), dtype="O")
        for i in range(len(embeddings)):
            vectors[i] = np.array(embeddings[i], copy=False)
        self.vector_index.update_batch(
            vectors=vectors,
            external_ids=external_ids,
            timestamp=timestamp if timestamp != 0 else None,
        )
        docs = {}
        docs["text"] = np.array(texts)
        if metadatas is not None:
            metadata_attr = np.empty([len(metadatas)], dtype=object)
            i = 0
            for metadata in metadatas:
                metadata_attr[i] = np.frombuffer(pickle.dumps(metadata), dtype=np.uint8)
                i += 1
            docs["metadata"] = metadata_attr
        docs_array = tiledb.open(
            self.docs_array_uri,
            "w",
            timestamp=timestamp if timestamp != 0 else None,
            config=self.config,
        )
        docs_array[external_ids] = docs
        docs_array.close()
        return ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_uri: str = "/tmp/tiledb_array",
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> TileDB:
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            metric=metric,
            index_uri=index_uri,
            index_type=index_type,
            config=config,
            index_timestamp=index_timestamp,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        index_uri: str,
        *,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> TileDB:
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        return cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            metric=metric,
            index_uri=index_uri,
            index_type=index_type,
            config=config,
            index_timestamp=index_timestamp,
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        index_uri: str,
        embedding: Embeddings,
        *,
        metric: str = DEFAULT_METRIC,
        config: Optional[Mapping[str, Any]] = None,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> TileDB:
        return cls(
            embedding=embedding,
            index_uri=index_uri,
            metric=metric,
            config=config,
            timestamp=timestamp,
            **kwargs,
        )

    def consolidate_updates(self, **kwargs: Any) -> None:
        self.vector_index = self.vector_index.consolidate_updates(**kwargs)
