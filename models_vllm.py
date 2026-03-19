from abc import ABC, abstractmethod
from typing import Optional
import torch
import json
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class _NGramRequestProcessor:
    def __init__(self, forbidden_ngrams: dict, n: int, penalty: float):
        self.forbidden_ngrams = forbidden_ngrams
        self.n = n
        self.penalty = penalty

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if len(output_ids) < self.n - 1:
            return logits
        last_tokens = tuple(output_ids[-(self.n - 1):])
        if last_tokens in self.forbidden_ngrams:
            for tok in self.forbidden_ngrams[last_tokens]:
                logits[tok] -= self.penalty
        return logits


class NGramAdapterLogitsProcessor(AdapterLogitsProcessor):

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra = params.extra_args or {}
        if "forbidden_ngrams_json" in extra:
            try:
                json.loads(extra["forbidden_ngrams_json"])
            except (json.JSONDecodeError, TypeError):
                raise ValueError("forbidden_ngrams_json geçerli bir JSON değil")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        extra = params.extra_args or {}
        ngrams_json = extra.get("forbidden_ngrams_json")
        n = extra.get("ngram_n", 3)
        penalty = extra.get("ngram_penalty", 5.0)

        if not ngrams_json:
            return None

        raw = json.loads(ngrams_json)
        forbidden_ngrams = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in raw.items()
            if k
        }

        return _NGramRequestProcessor(forbidden_ngrams, n=n, penalty=penalty)


class LModel(ABC):
    def __init__(self, model_id: str, **llm_kwargs):
        self.model_id = model_id
        self.model = LLM(
            model=model_id,
            download_dir="./cache",
            dtype="auto",
            max_model_len=4096,
            max_num_seqs=32,
            gpu_memory_utilization=0.85,
            logits_processors=[NGramAdapterLogitsProcessor],
            **llm_kwargs,
        )

    @abstractmethod
    def generate(
        self,
        input_str: str,
        sys_prompt: str,
        temperature: float,
        forbidden_ngram_map: dict,
    ) -> tuple[str, list[int]]:
        raise NotImplementedError(f"Generate not implemented for {self.model_id}")


class QwenLModel(LModel):
    def __init__(self, model_id: str, n_size: int = 3, penalty: float = 5.0, num_generations: int = 8, **llm_kwargs):
        super().__init__(model_id, **llm_kwargs)
        self.n_size = n_size
        self.penalty = penalty
        self.num_generations = num_generations 

    def generate(
        self,
        input_str: str,
        sys_prompt: str,
        temperature: float,
        forbidden_ngram_map: dict,
    ) -> tuple[str, list[int]]:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": input_str},
        ]

        tokenizer = self.model.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        ngrams_json = json.dumps({
            ",".join(str(t) for t in k): v
            for k, v in forbidden_ngram_map.items()
            if len(k) > 0
        }) if forbidden_ngram_map else None

        extra_args: dict = {
            "ngram_n": self.n_size,
            "ngram_penalty": self.penalty,
        }
        if ngrams_json:
            extra_args["forbidden_ngrams_json"] = ngrams_json

        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=temperature,
            extra_args=extra_args,
        )

        outputs = self.model.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        result = outputs[0].outputs[0]
        return result.text, list(result.token_ids)
        
    def generate_parallel(
        self,
        questions: list[str],
        sys_prompt: str,
        temperature: float,
    ) -> list[tuple[list[str], list[list[int]]]]:
        """
        Birden fazla soruyu aynı anda işle, her sorunun generation'ları sequential.
        Return: [(answers, token_ids_list), ...] her soru için
        """
        tokenizer = self.model.get_tokenizer()

        # Her soru için state tut
        states = [
            {
                "question": q,
                "forbidden_ngrams": {},
                "answers": [],
                "token_ids_list": [],
            }
            for q in questions
        ]

        for gen_idx in range(self.num_generations):
            # Bu generation round'u için tüm sorulara ait prompt + params listesi
            prompts = []
            sampling_params_list = []

            for state in states:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": state["question"]},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

                # Her sorunun kendi forbidden_ngram_map'i
                ngrams_json = json.dumps({
                    ",".join(str(t) for t in k): v
                    for k, v in state["forbidden_ngrams"].items()
                    if k
                }) if state["forbidden_ngrams"] else None

                extra_args = {
                    "ngram_n": self.n_size,
                    "ngram_penalty": self.penalty,
                }
                if ngrams_json:
                    extra_args["forbidden_ngrams_json"] = ngrams_json

                sampling_params_list.append(SamplingParams(
                    max_tokens=512,
                    temperature=temperature,
                    extra_args=extra_args,
                ))

            # Tüm sorular için tek batch generate
            outputs = self.model.generate(
                prompts=prompts,
                sampling_params=sampling_params_list,  # her prompt'a ayrı params
                use_tqdm=False,
            )

            # Sonuçları state'lere yaz, n-gram'ları güncelle
            for state, output in zip(states, outputs):
                result = output.outputs[0]
                content = result.text
                token_ids = list(result.token_ids)

                state["answers"].append(content)
                state["token_ids_list"].append(token_ids)

                # N-gram güncelle
                n = self.n_size
                if len(token_ids) >= n:
                    for j in range(len(token_ids) - n + 1):
                        prefix = tuple(token_ids[j: j + n - 1])
                        target = token_ids[j + n - 1]
                        if prefix not in state["forbidden_ngrams"]:
                            state["forbidden_ngrams"][prefix] = []
                        if target not in state["forbidden_ngrams"][prefix]:
                            state["forbidden_ngrams"][prefix].append(target)

        return [(s["answers"], s["token_ids_list"]) for s in states]