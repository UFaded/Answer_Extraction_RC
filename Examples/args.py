import torch

seed = 42
device = torch.device("cuda", 0)
test_lines = 187818  # 多少条训练数据，即：len(features)

train_set = "../Dataset/small_train_data.json"

max_seq_length = 512
max_query_length = 60

output_dir = "./model_dir"
predict_example_files='predict.data'
