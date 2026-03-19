import re

def parse_answer(example):
      text = example['answer']

      # Regex: last integer or float
      match = re.search(r'(\d+(?:\.\d+)?)(?!.*\d)', text)

      if match:
          answer = float(match.group(1))
      else:
          answer = None

      return {
          "question": example["question"],
          "answer": answer,
          "description": example["answer"]
      }