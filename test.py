from GPT2.encoder import get_encoder
import numpy as np
encoder = get_encoder()
batch_input_ids = []
file_path = 'testdata.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        text = line.strip()
        if not text:
            continue
        input_ids = encoder.encode(text)
        batch_input_ids.append(input_ids)
        print("text:", text)
        # 换行
        print("")
        print(batch_input_ids)
        print("")