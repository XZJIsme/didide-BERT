import torch
import pickle
import tqdm
from transformers import BertTokenizer
import didide_model
import optim_schedule
import random
import os


bert_path = "BERT-trained/chinese_L-12_H-768_A-12"
pretrained_model_path = bert_path
vocab_path = os.path.join(bert_path, "vocab.txt")
samples_path = "data/samples/wiki_zh.pkl"
save_path = "didide_model.pt"

tokenizer = BertTokenizer.from_pretrained(vocab_path)
# more recommended way is BertTokenizer.from_pretrained('bert_path')


class DiDiDeDataset(torch.utils.data.Dataset):
    def __init__(self, samples_path="data/samples/wiki_zh_mini.pkl", shuffle=True):
        tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.didide_sentences = []
        with open(samples_path, 'rb') as f:
            self.didide_sentences = pickle.load(f)
            # shuffle
            if shuffle:
                random.shuffle(self.didide_sentences)

    def __len__(self):
        return len(self.didide_sentences)

    def __getitem__(self, idx):
        return self.didide_sentences[idx]


dididedataset = DiDiDeDataset()
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dididedataset, [int(len(dididedataset)*0.8), int(len(dididedataset)*0.1), len(dididedataset)-int(len(dididedataset)*0.8)-int(len(dididedataset)*0.1)])

batch_size = 1024 + 256
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


model = didide_model.DiDiDeModelClass(
    pretrained_model_path=pretrained_model_path, dropout_rate=0.0)
hidden_size = model.bert.config.hidden_size

if torch.cuda.is_available():
    model.cuda(0)
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

# optimizer = torch.optim.Adam(
#     model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(
#     model.parameters(), lr=1e-5, momentum=0.9)
# adam_optimizer_scheduler = optim_schedule.ScheduledOptim(
#     optimizer, hidden_size, n_warmup_steps=10000)


# def loss_fn(outputs, targets):
#     # ToDo: check which loss to use
#     # return torch.nn.functional.cross_entropy(outputs, targets)
#     # return torch.nn.BCEWithLogitsLoss()(outputs, targets)
#     # use cross entropy loss
#     return torch.nn.functional.cross_entropy(outputs, targets)

# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

epochs = 3

for epoch in range(epochs):
    model.train()
    for i, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        tokenized = tokenizer(
            batch[0], batch[1], padding='longest')
        for key in tokenized.keys():
            tokenized[key] = torch.tensor(tokenized[key]).to(device)
        targets = torch.nn.functional.one_hot(
            batch[2], num_classes=3).to(device)
        targets = targets.float()
        outputs = model(**tokenized)
        loss = loss_fn(outputs, targets)
        if i % 10 == 0:
            print("epoch: {}, step: {}, loss: {}".format(epoch, i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # adam_optimizer_scheduler.step_and_update_lr()

    accuracy = 0
    model.eval()
    val_data_iter = tqdm.tqdm(
        enumerate(val_dataloader), total=len(val_dataloader))
    for i, batch in val_data_iter:
        tokenized = tokenizer(
            batch[0], batch[1], padding='longest')
        for key in tokenized.keys():
            tokenized[key] = torch.tensor(tokenized[key]).to(device)
        targets = torch.nn.functional.one_hot(
            batch[2], num_classes=3).to(device)
        targets = targets.float()
        outputs = model(**tokenized)
        accuracy += torch.sum(torch.argmax(outputs, dim=-1)
                              == torch.argmax(targets, dim=-1)).item()
    accuracy /= len(val_dataset)
    print("epoch: {}, accuracy: {}".format(epoch, accuracy))

model.eval()
test_data_iter = tqdm.tqdm(
    enumerate(test_dataloader), total=len(test_dataloader))
accuracy = 0
loss = 0.0
for i, batch in test_data_iter:
    tokenized = tokenizer(
        batch[0], batch[1], padding='longest')
    for key in tokenized.keys():
        tokenized[key] = torch.tensor(tokenized[key]).to(device)
    targets = torch.nn.functional.one_hot(
        batch[2], num_classes=3).to(device)
    targets = targets.float()
    outputs = model(**tokenized)
    accuracy += torch.sum(torch.argmax(outputs, dim=-1)
                          == torch.argmax(targets, dim=-1)).item()
    loss += loss_fn(outputs, targets).item()
accuracy /= len(test_dataset)
loss /= len(test_dataset)
print("test accuracy: {}, loss: {}".format(accuracy, loss))

# save model's weights
if torch.cuda.device_count() > 1:
    torch.save(model.module.cpu(), save_path)
else:
    torch.save(model.cpu(), save_path)
