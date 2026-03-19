from utils import parse_answer
from models_vllm import QwenLModel
from saver import Saver
from datasets import load_dataset
from tqdm import tqdm
import argparse

cache = "cache"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ytu-ce-cosmos/gsm8k_tr")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n_size", type=int, default=3)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prefix", type=str, default="sft_data")
    parser.add_argument("--chunk_size", type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()

    raw_dataset = load_dataset(args.dataset, split="train", cache_dir=cache)
    dataset = raw_dataset.map(parse_answer)

    num_samples = args.num_samples

    SYS_PROMPT = """
    Sana verilen soruyu YALNIZCA Türkçe yanıtla. Başka hiçbir dil kullanma.
    Yanıtına geçmeden önce problemi analiz et ve çözüm için adım adım mantık yürüt.
    Çözüm sürecini mantıksal bir sırayla açıkla.
    Akıl yürütme bittiğinde, ulaştığın kesin sayısal sonucu YALNIZCA şu format ile ver:
    <answer>SADECE_SAYI</answer>

    Önemli: <answer> etiketinin içine yalnızca sayı yaz, başka hiçbir şey ekleme.
    """

    dataset_subset = dataset.select(range(num_samples))

    saver = Saver(
        columns=["question", "ground_truth", "generated_answers"],
        total_samples=num_samples,
        save_interval=args.save_interval,
        file_prefix=args.prefix,
    )

    qwen = QwenLModel(
        args.model_id,
        n_size=args.n_size,
        penalty=args.penalty,
        num_generations=args.num_generations,
    )

    chunk_size = args.chunk_size
    outer_bar = tqdm(
        range(0, num_samples, chunk_size),
        desc="Chunks",
        total=(num_samples + chunk_size - 1) // chunk_size,
    )

    for chunk_start in outer_bar:
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk = dataset_subset.select(range(chunk_start, chunk_end))

        questions = [row["question"] for row in chunk]

        results = qwen.generate_parallel(
            questions=questions,
            sys_prompt=SYS_PROMPT,
            temperature=args.temperature,
        )

        batch = [
            {
                "question": row["question"],
                "ground_truth": row["answer"],
                "generated_answers": answers,
            }
            for row, (answers, _) in zip(chunk, results)
        ]
        saver.add_batch(batch)  # chunk bitince toplu kaydet

        outer_bar.set_postfix({
            "işlenen": chunk_end,
            "kalan": num_samples - chunk_end,
        })

    saver.save()
if __name__ == "__main__":
    main()