from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import torch
from typing import Optional

from model import LMModule

class TextSampler:
    def __init__(
        self,
        lm: LMModule,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        self.lm = lm
        self.tok = tokenizer
        self.device = device
        
        # sampling defaults
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_k          = top_k
        self.top_p          = top_p

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float]   = None,
        top_k: Optional[int]           = None,
        top_p: Optional[float]         = None,
    ) -> str:
        """
        Token‐string wrapper around LMModule.sample.
        """
        # use either passed‐in or default
        T = max_new_tokens or self.max_new_tokens
        temp = temperature   or self.temperature
        k    = top_k         if top_k is not None else self.top_k
        p    = top_p         if top_p is not None else self.top_p

        # tokenize prompt
        inputs = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        # sample continuations as token IDs
        out_ids = self.lm.sample(
            input_ids     = inputs,
            max_new_tokens= T,
            temperature   = temp,
            top_k         = k,
            top_p         = p,
            eos_token_id  = self.lm.eos_token_id,
        )

        # drop the prompt tokens and decode the rest
        gen_ids = out_ids[0, inputs.shape[-1] :].tolist()
        return self.tok.decode(gen_ids, skip_special_tokens=True)