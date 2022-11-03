import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import f1_score
from clarification.codes.config import *


class Policy:
    def __init__(self, mode='TRAIN'):
        self.policy_threshold = 0.15
        if mode == 'TRAIN':
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            self.model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, num_labels=2)
            self.tokenizer.add_tokens(['[SEP]', '[ISEP]', '[DSEP]'])
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif mode == 'PREDICT':
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            self.model = torch.load(BERT_FINE_TUNED_PATH, map_location=torch.device('cpu'))
            self.tokenizer.add_tokens(['[SEP]', '[ISEP]', '[DSEP]'])
            self.model.resize_token_embeddings(len(self.tokenizer))

    def build_training_dataset(self):
        with open(POLICY_TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]
        texts, labels = [], []
        for line in lines:
            text, label = line.split('\t')
            texts.append(text)
            labels.append(int(label))
        for i in range(10):
            print(texts[i], labels[i])
        return texts, labels

    def model_training(self, X_train, y_train, epochs=5, batch_size=8):
        print(len(X_train), len(y_train))
        padding_len = 512
        tokens_train = self.tokenizer.batch_encode_plus(X_train, max_length=padding_len, pad_to_max_length=True,
                                                        truncation=True)

        print('converting dataset to pytorch tensor dataset')
        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(y_train)

        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        cross_entropy = torch.nn.CrossEntropyLoss()

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        for epoch in range(epochs):
            print(f"""Epoch {epoch}/{epochs} start""")
            self.model.train()
            total_loss = 0
            total_preds = []
            total_labels = []

            for step, batch in enumerate(train_dataloader):
                batch = [x.to(device) for x in batch]
                sent_id, mask, labels = batch

                self.model.zero_grad()
                preds = self.model(sent_id, mask).logits

                loss = cross_entropy(preds, labels)
                total_loss = total_loss + loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)
                total_labels.append(labels.detach().cpu().numpy())

                if step != 0 and step % 2000 == 0:
                    print(f"Epoch {epoch}/{epochs}, step:{step}, train_loss:{loss}")

            total_preds = np.concatenate(total_preds, axis=0)
            total_labels = np.concatenate(total_labels, axis=0)

            avg_loss = total_loss / len(train_dataloader)
            preds = np.argmax(total_preds, axis=1)
            train_f1 = f1_score(total_labels, preds, average='micro')

            torch.save(self.model, BERT_FINE_TUNED_PATH + '_' + str(epoch + 1) + '_epochs.pkl')
            print(f"Epoch {epoch}/{epochs}, train_loss: {avg_loss}, train_acc:{train_f1}\n")

    def model_predict(self, text):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        padding_len = 512
        text = self.tokenizer([text], max_length=padding_len, padding=True, truncation=True)

        test_seq = torch.tensor(text['input_ids'])
        test_mask = torch.tensor(text['attention_mask'])
        pred = self.model(test_seq.to(device), test_mask.to(device)).logits
        pred = pred.detach().cpu().numpy()[0]

        def softmax(x):
            row_max = np.max(x)
            x -= row_max
            x_exp = np.exp(x)
            x_sum = np.sum(x_exp)
            s = x_exp / x_sum
            return s

        pred_softmax = softmax(pred)
        print(pred_softmax)
        true_prob = pred_softmax[1]
        if true_prob > self.policy_threshold:
            return True
        else:
            return False
