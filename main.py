import utils.http_session
import utils.transport
import utils.session
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VectorStore:

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        # Every embedding stored in this database must have this size.
        self.dimension = dimension

        # Distance metric used during similarity search.
        self.metric = metric

        # Maps vector IDs to their embeddings.
        self.vectors: Dict[str, np.ndarray] = {}

        # Stores additional information for each vector.
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Reserved for future indexing structures.
        self.index: Optional[List[str]] = None

    def _validate_vector(self, vector: List[float]) -> np.ndarray:
        """Convert a Python list into a NumPy array and verify its dimension."""
        array = np.array(vector, dtype=np.float32)

        if array.shape[0] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {array.shape[0]}"
            )

        return array

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return the distance between two vectors."""

        if self.metric == "cosine":
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 1.0

            cosine_similarity = np.dot(a, b) / (norm_a * norm_b)
            return 1.0 - cosine_similarity

        if self.metric == "euclidean":
            return float(np.linalg.norm(a - b))

        if self.metric == "dot":
            return -float(np.dot(a, b))

        raise ValueError(f"Unknown metric: {self.metric}")

    def insert(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a new vector."""
        self.vectors[vector_id] = self._validate_vector(vector)
        self.metadata[vector_id] = metadata or {}

    def search(
        self, query_vector: List[float], k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform an exact brute-force nearest-neighbor search."""
        query = self._validate_vector(query_vector)

        # Store (distance, vector_id) pairs so they can be sorted later.
        ranked_vectors: List[Tuple[float, str]] = []

        for vector_id, stored_vector in self.vectors.items():
            distance = self._distance(query, stored_vector)
            ranked_vectors.append((distance, vector_id))

        ranked_vectors.sort(key=lambda item: item[0])

        results = []
        for distance, vector_id in ranked_vectors[:k]:
            results.append(
                (
                    vector_id,
                    float(distance),
                    self.metadata.get(vector_id, {}),
                )
            )

        return results

    def search_batch(
        self, query_vector: List[float], k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Use NumPy vectorization to search faster."""

        if not self.vectors:
            return []

        query = self._validate_vector(query_vector)

        vector_ids = list(self.vectors.keys())
        matrix = np.stack([self.vectors[vid] for vid in vector_ids])

        if self.metric == "cosine":
            vector_norms = np.linalg.norm(matrix, axis=1)
            query_norm = np.linalg.norm(query)

            similarities = np.dot(matrix, query) / (vector_norms * query_norm)
            best_indices = np.argsort(-similarities)[:k]

            return [
                (
                    vector_ids[i],
                    float(1 - similarities[i]),
                    self.metadata.get(vector_ids[i], {}),
                )
                for i in best_indices
            ]

        if self.metric == "euclidean":
            distances = np.linalg.norm(matrix - query, axis=1)
            best_indices = np.argsort(distances)[:k]

            return [
                (
                    vector_ids[i],
                    float(distances[i]),
                    self.metadata.get(vector_ids[i], {}),
                )
                for i in best_indices
            ]

        if self.metric == "dot":
            scores = np.dot(matrix, query)
            best_indices = np.argsort(-scores)[:k]

            return [
                (
                    vector_ids[i],
                    float(-scores[i]),
                    self.metadata.get(vector_ids[i], {}),
                )
                for i in best_indices
            ]

        raise ValueError(f"Unknown metric: {self.metric}")

    def update(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing vector."""
        if vector_id not in self.vectors:
            return False

        self.vectors[vector_id] = self._validate_vector(vector)

        if metadata is not None:
            self.metadata[vector_id] = metadata

        return True

    def delete(self, vector_id: str) -> bool:
        """Remove a vector from the database."""
        if vector_id not in self.vectors:
            return False

        del self.vectors[vector_id]
        self.metadata.pop(vector_id, None)
        return True

    @property
    def count(self) -> int:
        """Return the number of stored vectors."""
        return len(self.vectors)

    def save(self, filepath: str) -> None:
        """Persist the database to disk."""

        data = {
            "dimension": self.dimension,
            "metric": self.metric,
            "vectors": {
                vector_id: vector.tolist()
                for vector_id, vector in self.vectors.items()
            },
            "metadata": self.metadata,
        }

        with open(filepath, "wb") as file:
            pickle.dump(data, file)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """Load a previously saved database."""

        with open(filepath, "rb") as file:
            data = pickle.load(file)

        database = cls(data["dimension"], data["metric"])

        for vector_id, vector in data["vectors"].items():
            database.vectors[vector_id] = np.array(vector, dtype=np.float32)

        database.metadata = data["metadata"]
        return database


if __name__ == "__main__":
    database = VectorStore(dimension=3, metric="cosine")

    database.insert("doc1", [1, 2, 3], {"title": "AI"})
    database.insert("doc2", [2, 3, 4], {"title": "Python"})
    database.insert("doc3", [8, 9, 10], {"title": "Vector DB"})

    print(f"Stored vectors: {database.count}")

    print("\nBrute-force search:")
    print(database.search([1, 2, 2.9], k=2))

    print("\nBatch search:")
    print(database.search_batch([1, 2, 2.9], k=2))

    database.save("vectors.pkl")

    restored = VectorStore.load("vectors.pkl")

    print("\nLoaded database:")
    print(restored.search([1, 2, 2.9], k=2))