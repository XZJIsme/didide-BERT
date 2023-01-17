import torch
import transformers
import logging
import os


class DiDiDeModelClass(torch.nn.Module):
    def __init__(self, pretrained_model_path, dropout_rate=0.1):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            pretrained_model_name_or_path=os.path.join(
                pretrained_model_path, "pytorch_model.bin"
            ),
            config=os.path.join(pretrained_model_path, "bert_config.json"),
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier1 = torch.nn.Linear(self.bert.config.hidden_size, 1024)
        self.classifier2 = torch.nn.Linear(1024, 3)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        output = output["pooler_output"]
        output = self.dropout(output)
        output = self.classifier1(output)
        output = self.classifier2(output)
        output = self.softmax(output)
        return output


if __name__ == "__main__":
    dididemodel = DiDiDeModelClass(
        pretrained_model_path="BERT-trained/chinese_L-12_H-768_A-12"
    )
