"""
Base Planner — LLM Abstraction Layer
=====================================

Provides BasePlanner ABC with concrete implementations for Gemini, OpenAI, and
fallback (no-LLM) planners. Factory function reads provider from config.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.utils.config import config

logger = logging.getLogger(__name__)

load_dotenv()


class BasePlanner(ABC):
    """Abstract interface for LLM-based strategy proposal generation."""

    @abstractmethod
    def generate_proposals(
        self,
        prompt: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Send a structured prompt to the LLM and parse JSON responses.

        Returns:
            List of dicts, each with at least: fast_window, slow_window,
            reasoning, confidence, regime_characteristic_used.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this planner has valid credentials."""
        ...


class GeminiPlanner(BasePlanner):
    """Google Gemini implementation via google-generativeai SDK."""

    def __init__(self):
        self._api_key = os.getenv("GOOGLE_API_KEY", "")
        self._model_name = config.llm.model
        self._temperature = config.llm.temperature
        self._model = None

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key not in ("", "your_gemini_api_key_here", "you api key"))

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config={"temperature": self._temperature},
            )
        return self._model

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        model = self._get_model()
        response = model.generate_content(prompt)

        text = response.text.strip()
        return self._parse_json_response(text, n)

    def _parse_json_response(self, text: str, n: int) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response text."""
        # Try to find JSON array in response
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                proposals = json.loads(text[start : end + 1])
                if isinstance(proposals, list):
                    return proposals[:n]
            except json.JSONDecodeError:
                pass

        # Try single JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return [obj]
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM response as JSON: %s...", text[:200])
        return []


class LangChainPlanner(BasePlanner):
    """LangChain + Gemini implementation for structured output."""

    def __init__(self):
        self._api_key = os.getenv("GOOGLE_API_KEY", "")
        self._model_name = config.llm.model
        self._temperature = config.llm.temperature

    def is_available(self) -> bool:
        if not self._api_key or self._api_key in ("", "your_gemini_api_key_here", "you api key"):
            return False
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401
            return True
        except ImportError:
            return False

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=self._temperature,
            max_retries=config.llm.max_retries,
        )
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return GeminiPlanner._parse_json_response(None, text, n)


class OpenAIPlanner(BasePlanner):
    """OpenAI implementation for users with an OpenAI key."""

    def __init__(self):
        self._api_key = os.getenv("OPENAI_API_KEY", "")

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=config.llm.temperature,
        )
        text = response.choices[0].message.content or ""
        return GeminiPlanner._parse_json_response(None, text, n)


class FallbackPlanner(BasePlanner):
    """No-LLM fallback — always returns empty (caller uses grid search)."""

    def is_available(self) -> bool:
        return True

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        logger.info("FallbackPlanner: no LLM available, returning empty proposals.")
        return []


def create_planner(provider: Optional[str] = None) -> BasePlanner:
    """
    Factory: create the appropriate planner based on config or override.

    Priority: LangChain Gemini > raw Gemini > OpenAI > Fallback.
    """
    provider = provider or config.llm.provider

    planners = {
        "gemini": [LangChainPlanner, GeminiPlanner],
        "openai": [OpenAIPlanner],
        "ollama": [FallbackPlanner],  # placeholder for future Ollama support
    }

    candidates = planners.get(provider, [GeminiPlanner, OpenAIPlanner])

    for planner_cls in candidates:
        planner = planner_cls()
        if planner.is_available():
            logger.info("Using %s as LLM planner.", planner_cls.__name__)
            return planner

    logger.warning("No LLM planner available. Using FallbackPlanner (grid search only).")
    return FallbackPlanner()
