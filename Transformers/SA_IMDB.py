import wget, tarfile
import os


# ----- download dataset -----
def download_dataset(url: str, save_path: str) -> None:
    extra_path = "data/" + save_path.split(".")[0]
    if not os.path.isdir(extra_path):
        wget.download(url, out=save_path)
        with tarfile.open(save_path) as tf:
            extra_path = "data/"+save_path.split(".")[0]
            tf.extractall(extra_path)
        os.remove(save_path)
        print("Download success in {}".format(extra_path))
    else:
        print("data has downloaded in {}".format(extra_path))


data_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
save_file = 'aclImdb_v1.tar.gz'
download_dataset(data_url, save_file)

# ---------- SA IMDB ----------
# ----- data -----
from pathlib import Path


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text(encoding="utf-8"))
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


train_texts, train_labels = read_imdb_split('data/aclImdb_v1/aclImdb/train')
test_texts, test_labels = read_imdb_split('data/aclImdb_v1/aclImdb/test')

# ----- split train valid -----
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# ----- BertTokenizer -----
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


# ----- Dataset -----
import torch


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)


# ----- Trainer -----
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import EvaluationStrategy

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=3,   # batch size per device during training
    per_device_eval_batch_size=3,    # batch size for evaluation
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_steps=50,
    evaluation_strategy=EvaluationStrategy.STEPS,
    eval_steps=250,
    gradient_accumulation_steps=12,
)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()


# predict
# ----- test data -----

prediction = trainer.predict(test_dataset)
print(prediction)
