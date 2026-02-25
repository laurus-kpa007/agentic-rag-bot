"""설정 관리 모듈 - .env 파일에서 설정을 로드한다."""

import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


class Config:
    def __init__(self):
        self.ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.llm_model: str = os.getenv("LLM_MODEL", "qwen3:14b")
        self.mcp_config_path: str = os.getenv("MCP_CONFIG_PATH", "mcp_config.json")
        self.chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "bona/bge-m3-korean:latest")
        self.hitl_mode: str = os.getenv("HITL_MODE", "auto")
        self.max_tool_calls: int = int(os.getenv("MAX_TOOL_CALLS", "5"))
        self.max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent
