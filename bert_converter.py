import transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch as bert_2_pytorch

# tf_checkpoint_path = "BERT-trained/multi_cased_L-12_H-768_A-12/bert_model.ckpt"
# bert_config_file = "BERT-trained/multi_cased_L-12_H-768_A-12/bert_config.json"
# pytorch_dump_path = "BERT-trained/multi_cased_L-12_H-768_A-12/pytorch_model.bin"

tf_checkpoint_path = "BERT-trained/chinese_L-12_H-768_A-12/bert_model.ckpt"
bert_config_file = "BERT-trained/chinese_L-12_H-768_A-12/bert_config.json"
pytorch_dump_path = "BERT-trained/chinese_L-12_H-768_A-12/pytorch_model.bin"

bert_2_pytorch.convert_tf_checkpoint_to_pytorch(
    tf_checkpoint_path, bert_config_file, pytorch_dump_path)
