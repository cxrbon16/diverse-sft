from utils import parse_answer
from models import QwenLModel
from saver import Saver

from datasets import load_dataset
from tqdm import tqdm

import argparse

cache = "cache"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ytu-ce-cosmos/gsm8k_tr")
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--num_generations", type=int, default=5)
parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--n_size", type=int, default=3)
parser.add_argument("--penalty", type=float, default=0.0)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--prefix", type=str, default="sft_data")

args = parser.parse_args()

raw_dataset = load_dataset(args.dataset, split="train", cache_dir=cache)
dataset = raw_dataset.map(parse_answer)

num_samples = args.num_samples
num_generations = args.num_generations

SYS_PROMPT = """
Sana verilen soruyu Türkçe yanıtla. Yanıtına geçmeden önce problemi analiz et ve çözüm için adım adım mantık yürüt (Chain-of-Thought).
Çözüm sürecini mantıksal bir sırayla açıkla. Akıl yürütme bittiğinde, ulaştığın kesin sonucu şu etiketi kullanarak mühürle:

<answer>"your answer here."</answer>
"""

# Veri setini seç (HuggingFace datasets varsayıldı)
dataset_subset = dataset.select(range(num_samples))
saver = Saver(columns=["question", "ground_truth", "generated_answers"], total_samples=num_samples, save_interval=args.save_interval, file_prefix=args.prefix)

# Modeli başlat
qwen = QwenLModel(args.model_id, n_size=args.n_size, penalty=args.penalty)

for index, row in tqdm(enumerate(dataset_subset), total=num_samples):
    # Her yeni soru için yasaklı n-gram havuzunu sıfırla
    session_forbidden_ngrams = {}
    q = row["question"]
    generated_answers = []

    for i in range(num_generations):
        content, token_ids = qwen.generate(
            input_str=q,
            sys_prompt=SYS_PROMPT,
            temperature=args.temperature,
            forbidden_ngram_map=session_forbidden_ngrams
        )

        # 2. Üretilen cevaptan yeni n-gram'ları çıkar ve havuza ekle
        n = qwen.n_size
        if len(token_ids) >= n:
            for j in range(len(token_ids) - n + 1):
                prefix = tuple(token_ids[j : j + n - 1])
                target = token_ids[j + n - 1]

                if prefix not in session_forbidden_ngrams:
                    session_forbidden_ngrams[prefix] = []

                if target not in session_forbidden_ngrams[prefix]:
                    session_forbidden_ngrams[prefix].append(target)
        generated_answers.append(content)

    # Veriyi kaydet
    saver.add({
        "question": q,
        "ground_truth": row["answer"],
        "generated_answers": generated_answers,
    })
