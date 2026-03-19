from typing import List
from datetime import datetime
import pandas as pd
import os

class Saver(object):
    def __init__(self,
                columns: List[str],
                total_samples: int,
                file_prefix: str = "sft_data",
                save_interval: int = 10,
                ):

        file_name = f"{file_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        os.makedirs("results", exist_ok=True)
        self.save_path = f"results/{file_name}"
        self.data = {
            columns[i]: [None for _ in range(total_samples)] for i in range(len(columns))
        } 
        self.total_samples = total_samples
        self.save_interval = save_interval
        self.columns = columns
        self.index = 0


    def add(self, item):
        for key in self.columns:
            self.data[key][self.index] = item[key]
        self.index += 1
        if self.index % self.save_interval == 0 or self.index == self.total_samples:
            self.save()
    
    def save(self):
        df = pd.DataFrame(self.data, columns=self.columns)
        df.to_csv(self.save_path, index=False)
