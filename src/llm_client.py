"""
llm_client.py
-------------
Generation backend. Two free options, tried in order:

  1. Groq (config.GROQ_API_KEY) - fast, generous free tier, good open
     models (Llama 3.x). Preferred when available.
  2. Hugging Face Inference API (config.HF_API_TOKEN) - free-tier fallback.

Both are wrapped behind `LLMClient.generate(system, user)` so
`rag_pipeline.py` doesn't care which backend answered.
"""
from __future__ import annotations

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        groq_api_key: str = config.GROQ_API_KEY,
        groq_model: str = config.GROQ_MODEL,
        hf_api_token: str = config.HF_API_TOKEN,
        hf_model: str = config.HF_MODEL,
    ):
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.hf_api_token = hf_api_token
        self.hf_model = hf_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.groq_api_key:
            try:
                return self._generate_groq(system_prompt, user_prompt)
            except Exception as e:
                logger.warning("Groq generation failed (%s), trying HF fallback", e)

        if self.hf_api_token:
            try:
                return self._generate_hf(system_prompt, user_prompt)
            except Exception as e:
                logger.error("HF fallback generation also failed: %s", e)

        return (
            "No LLM backend is configured or reachable. Set GROQ_API_KEY "
            "(https://console.groq.com) or HF_API_TOKEN as environment "
            "variables to enable answer generation."
        )

    # ------------------------------------------------------------------
    def _generate_groq(self, system_prompt: str, user_prompt: str) -> str:
        from groq import Groq

        client = Groq(api_key=self.groq_api_key)
        resp = client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.GENERATION_TEMPERATURE,
            max_tokens=config.GENERATION_MAX_TOKENS,
        )
        return resp.choices[0].message.content

    def _generate_hf(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        payload = {
            "inputs": f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n",
            "parameters": {
                "temperature": config.GENERATION_TEMPERATURE,
                "max_new_tokens": config.GENERATION_MAX_TOKENS,
                "return_full_text": False,
            },
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
