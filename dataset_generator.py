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
for file in tqdm.tqdm(files[:15]):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            text = json_data['text']
            for i in range(len(text)):
                if text[i] == '的' or text[i] == '地' or text[i] == '得':
                    if text[i] == '的':
                        l_ = random.randint(1, 15)
                        r_ = random.randint(1, 15)
                        sample = (text[i-l_:i], text[i+1:i+r_],
                                dict_dedede[text[i]])
                        samples.append(sample)
                    else:
                        for j in range(random.randint(1, 15)):
                            l_ = random.randint(1, 15)
                            r_ = random.randint(1, 15)
                            sample = (text[i-l_:i], text[i+1:i+r_],
                                    dict_dedede[text[i]])
                            samples.append(sample)

print("Total samples: ", len(samples))

的 = 0
地 = 0
得 = 0
for sample in samples:
    if sample[2] == 0:
        的 += 1
    elif sample[2] == 1:
        地 += 1
    else:
        得 += 1

print("的：{}，地：{}，得：{}".format(的, 地, 得))

with open('data/samples/wiki_zh_mini.pkl', 'wb+') as f:
    pickle.dump(samples, f)
    