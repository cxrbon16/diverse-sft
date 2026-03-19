from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

class PreviousNGramTaskLogitsProcessor(nn.Module):
    def __init__(self, n: int, forbidden_ngrams: set, penalty: float = 5.0):
        super().__init__()
        self.n = n
        self.forbidden_ngrams = forbidden_ngrams # {(token1, token2): [yasakli_token3, yasakli_token4]}
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Mevcut üretimin son (n-1) token'ına bakıyoruz
        if input_ids.shape[-1] < self.n - 1:
            return scores

        # Son n-1 token'ı al (tuple yap ki hashable olsun)
        last_tokens = tuple(input_ids[0, -(self.n - 1):].tolist())

        # Eğer bu başlangıç daha önce yasaklı bir n-gram oluşturduysa, devamını banla
        if last_tokens in self.forbidden_ngrams:
            tokens_to_ban = self.forbidden_ngrams[last_tokens]
            scores[0, tokens_to_ban] -= self.penalty

        return scores

class LModel(ABC):
    def __init__(self, model_id: str, penalty=5.0):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir = "./cache",
            torch_dtype="auto",
            device_map="auto"
        )

    @abstractmethod
    def generate(self, input_str: str, sys_prompt: str, temperature: float, forbidden_ngram_map: dict):
        raise NotImplementedError(f"Generate not implemented for {self.model_id}")

class QwenLModel(LModel):
  def __init__(self, model_id: str, n_size: int = 3, penalty: float = 5.0):
      super().__init__(model_id)
      self.n_size = n_size # Kaçlı n-gram takibi yapılacağı (Trigram için 3)
      self.penalty = penalty

  def generate(self, input_str: str, sys_prompt: str, temperature: float, forbidden_ngram_map: dict) -> str:
      messages = [
          {"role": "system", "content": sys_prompt},
          {"role": "user", "content": input_str}
      ]

      text = self.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True,
      )

      model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

      # n_size ve yasaklı n-gram haritasını işlemciye gönderiyoruz
      ngram_processor = PreviousNGramTaskLogitsProcessor(
          n=self.n_size,
          forbidden_ngrams=forbidden_ngram_map,
          penalty = self.penalty
      )

      generated_ids = self.model.generate(
          **model_inputs,
          max_new_tokens=512,
          do_sample=True, # Çeşitlilik için sampling açık olmalı
          temperature=temperature,
          logits_processor=LogitsProcessorList([ngram_processor])
      )

      # Sadece yeni üretilen tokenları al
      input_length = model_inputs.input_ids.shape[1]
      output_ids = generated_ids[0][input_length:].tolist()

      content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
      return content, output_ids # Hem metni hem token listesini dönüyoruz ki havuzu güncelleyelim
