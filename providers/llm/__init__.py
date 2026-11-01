"""Optional LLM providers. The pipeline works without one (template/heuristic paths); an LLM only
improves script prose and metadata wording. All LLM output is validated against research/transcript."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from abc import ABC, abstractmethod


class LLMUnavailable(RuntimeError):
    pass


class LLMProvider(ABC):
    name = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        ...

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        ...


class CommandLLM(LLMProvider):
    """Pipe the prompt to any local command (e.g. llama.cpp `llama-cli`, a wrapper script)."""
    name = "command"

    def available(self):
        cmd = self.config.get("command")
        if not cmd:
            return False, "llm.command not configured"
        return (True, "ok") if shutil.which(cmd[0]) else (False, f"{cmd[0]} not found")

    def complete(self, prompt, system=None, max_tokens=2048):
        ok, why = self.available()
        if not ok:
            raise LLMUnavailable(why)
        full = (system + "\n\n" if system else "") + prompt
        cp = subprocess.run(self.config["command"], input=full, capture_output=True, text=True,
                            encoding="utf-8", timeout=self.config.get("timeout", 600))
        if cp.returncode != 0:
            raise LLMUnavailable(f"LLM command failed: {cp.stderr.strip()[:200]}")
        return cp.stdout.strip()


class OllamaLLM(LLMProvider):
    name = "ollama"

    def available(self):
        return (True, "ok (server reachability checked at call time)") if self.config.get("model") else \
            (False, "llm.model not configured")

    def complete(self, prompt, system=None, max_tokens=2048):
        url = self.config.get("url", "http://127.0.0.1:11434") + "/api/generate"
        body = json.dumps({"model": self.config["model"], "prompt": prompt, "system": system or "",
                           "stream": False, "options": {"num_predict": max_tokens}}).encode()
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.config.get("timeout", 600)) as r:
                return json.loads(r.read())["response"].strip()
        except Exception as e:
            raise LLMUnavailable(f"ollama request failed: {e}") from e


class OpenAICompatLLM(LLMProvider):
    """Any OpenAI-compatible /v1/chat/completions endpoint. API key is read from an env var, never stored."""
    name = "openai_compatible"

    def available(self):
        if not self.config.get("base_url") or not self.config.get("model"):
            return False, "llm.base_url and llm.model required"
        env = self.config.get("api_key_env", "LLM_API_KEY")
        if not os.environ.get(env) and not self.config.get("no_key"):
            return False, f"environment variable {env} not set"
        return True, "ok"

    def complete(self, prompt, system=None, max_tokens=2048):
        ok, why = self.available()
        if not ok:
            raise LLMUnavailable(why)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = json.dumps({"model": self.config["model"], "messages": msgs, "max_tokens": max_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.get("api_key_env", "LLM_API_KEY"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            req = urllib.request.Request(self.config["base_url"].rstrip("/") + "/chat/completions", data=body,
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=self.config.get("timeout", 120)) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMUnavailable(f"LLM request failed: {e}") from e


REGISTRY = {"command": CommandLLM, "ollama": OllamaLLM, "openai_compatible": OpenAICompatLLM}


def get_llm(cfg: dict | None) -> LLMProvider | None:
    """Return a provider or None when llm.provider is 'none'/unset."""
    cfg = cfg or {}
    name = cfg.get("provider", "none")
    if name in (None, "none", ""):
        return None
    if name not in REGISTRY:
        raise KeyError(f"unknown llm provider {name!r}; options: {sorted(REGISTRY)} or 'none'")
    return REGISTRY[name](cfg)
