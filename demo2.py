import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from model import FunASRNano


class MaxConsecutiveNgramProcessor(LogitsProcessor):
    def __init__(self, min_ngram_size=2, max_ngram_size=8, max_consecutive=2):
        if min_ngram_size < 1:
            raise ValueError("min_ngram_size must be >= 1")
        if max_ngram_size < min_ngram_size:
            raise ValueError("max_ngram_size must be >= min_ngram_size")
        if max_consecutive < 1:
            raise ValueError("max_consecutive must be >= 1")

        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size
        self.max_consecutive = max_consecutive

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()
            banned = set()
            max_len = min(self.max_ngram_size, len(seq) // self.max_consecutive)
            for n in range(self.min_ngram_size, max_len + 1):
                tail_len = n * self.max_consecutive
                tail = seq[-tail_len:]
                if len(tail) < tail_len:
                    continue
                pattern = tail[:n]
                if all(tail[j * n : (j + 1) * n] == pattern for j in range(1, self.max_consecutive)):
                    banned.add(pattern[0])

            if banned:
                scores[i, list(banned)] = -float("inf")

        return scores


def main():
    # model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    model_dir = "/root/autodl-tmp/ML/mixed1/outputs"
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device=device)
    m.eval()

    tokenizer = kwargs.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("tokenizer is required for logits_processor")

    kwargs["hotwords"] = ["开放时间"]
    kwargs["max_length"] = 128
    logits_processor = LogitsProcessorList(
        [MaxConsecutiveNgramProcessor(min_ngram_size=2, max_ngram_size=8, max_consecutive=2)]
    )
    kwargs["llm_kwargs"] = {
        "repetition_penalty": 1.0,
        # "no_repeat_ngram_size": 100,
        # "logits_processor": logits_processor,
    }

    # wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    # wav_path = f"20260128-114412.mp3" # 大语言模型，间隔
    # wav_path = f"20260128-114849.mp3" # 大语言模型，连续
    wav_path = f"02a5601c02e4aa8f42c0e21549424d7eeb9f872f6fcf9091b3f5faa0d7a3246d.ogg"
    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
