"""OllamaEmbedder 단위 테스트"""

import numpy as np
from unittest.mock import patch, MagicMock
from src.embedding import OllamaEmbedder


class TestOllamaEmbedder:
    def _mock_response(self, embeddings):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": embeddings}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("src.embedding.requests.post")
    def test_encode_single_string(self, mock_post):
        """단일 문자열 → (dim,) 형태 반환."""
        mock_post.return_value = self._mock_response([[0.1, 0.2, 0.3]])

        embedder = OllamaEmbedder(model="bona/bge-m3-korean:latest")
        result = embedder.encode("테스트 문장")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        mock_post.assert_called_once()
        call_json = mock_post.call_args[1]["json"]
        assert call_json["model"] == "bona/bge-m3-korean:latest"
        assert call_json["input"] == ["테스트 문장"]

    @patch("src.embedding.requests.post")
    def test_encode_list(self, mock_post):
        """문자열 리스트 → (N, dim) 형태 반환."""
        mock_post.return_value = self._mock_response([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])

        embedder = OllamaEmbedder(model="bona/bge-m3-korean:latest")
        result = embedder.encode(["문장1", "문장2"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    @patch("src.embedding.requests.post")
    def test_encode_uses_correct_url(self, mock_post):
        """올바른 Ollama API URL 호출."""
        mock_post.return_value = self._mock_response([[0.1]])

        embedder = OllamaEmbedder(
            model="test-model",
            base_url="http://custom:1234",
        )
        embedder.encode("테스트")

        call_url = mock_post.call_args[0][0]
        assert call_url == "http://custom:1234/api/embed"

    @patch("src.embedding.requests.post")
    def test_encode_returns_float32(self, mock_post):
        """반환 타입이 float32."""
        mock_post.return_value = self._mock_response([[0.1, 0.2]])

        embedder = OllamaEmbedder()
        result = embedder.encode("테스트")
        assert result.dtype == np.float32
