"""Ollama 임베딩 어댑터

sentence-transformers 대신 Ollama /api/embed 엔드포인트를 사용한다.
SentenceTransformer와 동일한 인터페이스(encode)를 제공하여 기존 코드 호환성을 유지한다.
"""

import os

import numpy as np
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "bona/bge-m3-korean:latest")


class OllamaEmbedder:
    """Ollama /api/embed를 사용하는 임베딩 클래스.

    SentenceTransformer.encode()와 동일한 시그니처를 지원한다.
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def encode(self, texts, **kwargs) -> np.ndarray:
        """텍스트(문자열 또는 리스트)를 임베딩 벡터로 변환한다.

        Args:
            texts: 단일 문자열 또는 문자열 리스트
        Returns:
            np.ndarray: (N, dim) 형태의 임베딩 배열 (리스트 입력)
                        또는 (dim,) 형태 (단일 문자열 입력)
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = np.array(data["embeddings"], dtype=np.float32)

        if single:
            return embeddings[0]
        return embeddings
