import sys
import torch
import transformers
from transformers import BertTokenizer
import random, os
from didide_model import DiDiDeModelClass

model_path = "didide_model.pt"

bert_path = "BERT-trained/chinese_L-12_H-768_A-12"
# vocab_path = os.path.join(bert_path, "vocab.txt")
tokenizer = BertTokenizer.from_pretrained(bert_path)

if __name__ == "__main__":
    try:
        text = sys.argv[1]
    except:
        text = "我觉的我烦的有点难过，因为我得培根忘记吃了"
    try:
        model_path = sys.argv[2]
    except:
        pass
    # model is a DiDiDeModelClass
    model: DiDiDeModelClass = torch.load(model_path)
    model.eval()
    dict_dedede = {"的": 0, "地": 1, "得": 2}
    dedede_dict = {0: "的", 1: "地", 2: "得"}
    for i in range(len(text)):
        if text[i] == "的" or text[i] == "地" or text[i] == "得":
            opt = None
            for j in range(10):
                l_ = random.randint(5, 10)
                r_ = random.randint(5, 10)
                sample = (text[i - l_ : i], text[i + 1 : i + r_], dict_dedede[text[i]])
                tokenized_sample = tokenizer(sample[0], sample[1], padding="longest")
                for key in tokenized_sample.keys():
                    tokenized_sample[key] = [tokenized_sample[key]]
                for key in tokenized_sample.keys():
                    tokenized_sample[key] = torch.tensor(tokenized_sample[key])
                output = model(**tokenized_sample)
                if opt is None:
                    opt = output
                else:
                    opt += output
            opt = int(opt.argmax(dim=-1).item())
            text = text[:i] + dedede_dict[opt] + text[i + 1 :]

    print(text)
