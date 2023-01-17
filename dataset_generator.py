import glob
import os
import json
import random
import pickle
import tqdm
import numpy as np

dict_dedede = {'的': 0, '地': 1, '得': 2}

files = []
for file in glob.glob("data/wiki_zh/**", recursive=True):
    if os.path.isdir(file):
        continue
    files.append(file)

np.random.shuffle(files)

# read json files
samples = []
for file in tqdm.tqdm(files[:1]):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            text = json_data['text']
            for i in range(len(text)):
                if text[i] == '的' or text[i] == '地' or text[i] == '得':
                    for j in range(random.randint(1, 2)):
                        l_ = random.randint(1, 15)
                        r_ = random.randint(1, 15)
                        sample = (text[i-l_:i], text[i+1:i+r_],
                                  dict_dedede[text[i]])
                        samples.append(sample)

print("Total samples: ", len(samples))

with open('data/samples/wiki_zh_mini_1.pkl', 'wb+') as f:
    pickle.dump(samples, f)
    