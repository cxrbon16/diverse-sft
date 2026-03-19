from typing import List
from datetime import datetime
import pandas as pd
import os

class Saver:
    def __init__(
        self,
        columns: List[str],
        total_samples: int,
        file_prefix: str = "sft_data",
        save_interval: int = 10,
    ):
        file_name = f"{file_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        os.makedirs("results", exist_ok=True)
        self.save_path = f"results/{file_name}"
        self.columns = columns
        self.total_samples = total_samples
        self.save_interval = save_interval
        self.data = []       # artık dict listesi, pre-alloc yok
        self.index = 0

    def add(self, item: dict):
        self.data.append({key: item[key] for key in self.columns})
        self.index += 1
        if self.index % self.save_interval == 0 or self.index == self.total_samples:
            self.save()

    def add_batch(self, items: List[dict]):
        """Chunk sonunda toplu ekleme için."""
        for item in items:
            self.data.append({key: item[key] for key in self.columns})
        self.index += len(items)
        self.save()

    def save(self):
        df = pd.DataFrame(self.data, columns=self.columns)
        df.to_csv(self.save_path, index=False)
        print(f"[Saver] {self.index}/{self.total_samples} kayıt yazıldı → {self.save_path}")