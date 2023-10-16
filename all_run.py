import random
import torch
import pandas as pd
import numpy as np
import argparse
import time
import os
import csv
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformer_all import BertTokenizer, BertModel, BertConfig
from transformer_all import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from transformer_all import BertForSequenceClassification, AdamW
from transformer_all import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import spearmanr, pearsonr

class MyDataset():
    def __init__(self, args, path_to_file):
        self.dataset = pd.read_csv(path_to_file, encoding='utf-8', sep='\t')
        self.is_sim = args.is_sim
        self.pretrained_model = args.pretrained_model
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text_a = self.dataset.loc[idx, "text_a"]
        text_b = self.dataset.loc[idx, "text_b"]
        label = self.dataset.loc[idx, "labels"]
        #print(text_a, text_b)
        seq_length = int(128)
        if self.is_sim:
            similarity = self.dataset.loc[idx, "similarity"]
            similarity = np.array(eval(similarity))
            sim_matrix = np.zeros((seq_length, seq_length))
            if len(similarity) > seq_length:
                sim_matrix[:seq_length, :seq_length] = similarity[:seq_length, :seq_length]
            else:
                sim_matrix[:len(similarity), :len(similarity)] = similarity
            similarity = sim_matrix
        else:
            similarity=torch.tensor(0).to(device)
        encode_dict_result = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=128,
                                                   padding='max_length', return_attention_mask=True,
                                                   return_token_type_ids=True,
                                                   return_tensors='pt', truncation=True)

        input_ids = encode_dict_result["input_ids"].to(device)
        token_type_ids = encode_dict_result["token_type_ids"].to(device)
        attention_mask = encode_dict_result["attention_mask"].to(device)
        #print(attention_mask)
        if 'roberta' in self.pretrained_model:
            index = np.where(encode_dict_result["input_ids"].numpy()==2)[1]
            len_a = index[0]-1
            if len(index)==3:
                len_b = index[2]-index[1]-1
                len_ab = index[2]+1
            elif len(index)==2:
                len_b = 128 - index[1] -1
                len_ab = 128
            elif len(index)==1:
                len_b = 0
                len_ab = 128
            lens = [len_a, len_ab]
        else:
            if len(torch.unique(attention_mask,return_counts=True)[0])==2:
                len_ab = int(torch.unique(attention_mask,return_counts=True)[1][1])
                len_b = int(torch.unique(token_type_ids,return_counts=True)[1][1])-1
                len_a = len_ab - len_b -3
                lens = [len_a, len_ab]
            else:
                len_a = int(torch.unique(token_type_ids,return_counts=True)[1][0])-2
                len_b = int(torch.unique(token_type_ids,return_counts=True)[1][1])
                len_ab = 128
                lens = [len_a, len_ab]
                
        similarity = similarity/(len_ab)
        sample = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": label, "similarity": similarity, "lens": torch.tensor(lens, dtype=torch.long)}
        return sample

def evaluate(args, model, dataloader):
    model.eval()
    total_val_loss, total_eval_accuracy = 0, 0
    preds, golds = [], []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids, token_type_ids, attention_mask, lens = batch["input_ids"].squeeze(1), batch["token_type_ids"].squeeze(1), \
                                                        batch["attention_mask"].squeeze(1), batch["lens"]
            if args.is_regress:
                labels = batch["labels"].unsqueeze(1).float()
            else:
                labels = batch["labels"]
            similarity = batch["similarity"].to(device)
            if not args.is_sim:
                similarity = None
            output = model(input_ids, similarity=similarity, lens=lens, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=labels.to(device))
            loss, logits = output[0], output[1]
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)

            total_val_loss += loss.item()
            if args.is_regress:
                preds.extend(logits.squeeze(1).data.cpu().numpy())
                golds.extend(batch["labels"].data.cpu().numpy())
            else:
                logits = logits.detach().cpu().numpy()
                label_ids = batch["labels"].numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                preds.append(np.argmax(logits, axis=1).flatten())
                golds.append(label_ids.flatten())

    #计算得分
    if args.is_regress:
        scores = {}
        pear = pearsonr(preds, golds)[0]
        spea = spearmanr(preds, golds)[0]
        scores['pearsonr'] = pear
        scores['spearmanr'] = spea
        avg_val_loss = total_val_loss / len(dataloader)
        return avg_val_loss, 0, scores
    else:
        gold_labels = np.concatenate(golds)
        inference_labels = np.concatenate(preds)
        #计算得分
        scores = np.zeros((2,3))
        f11 = f1_score(gold_labels, inference_labels, average="binary", pos_label=1)
        p1 = precision_score(gold_labels, inference_labels, average="binary", pos_label=1)
        r1 = recall_score(gold_labels, inference_labels, average="binary", pos_label=1)

        f10 = f1_score(gold_labels, inference_labels, average="binary", pos_label=0)
        p0 = precision_score(gold_labels, inference_labels, average="binary", pos_label=0)
        r0 = recall_score(gold_labels, inference_labels, average="binary", pos_label=0)
        scores[0][0] = p0
        scores[0][1] = r0
        scores[0][2] = f10
        scores[1][0] = p1
        scores[1][1] = r1
        scores[1][2] = f11

        avg_val_loss = total_val_loss / len(dataloader)
        avg_val_accuracy = total_eval_accuracy / len(dataloader)
        return avg_val_loss, round(avg_val_accuracy, 4), scores


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    parser.add_argument("--task", type=str, default='sicksts')
    parser.add_argument("--task_name", type=str, default='layer')
    parser.add_argument("--pretrained_model", default='roberta-base', type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str)
    parser.add_argument("--train_corpus", type=str)
    parser.add_argument("--dev_corpus", type=str)
    parser.add_argument("--test_corpus", type=str)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_large", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3) #allrun
    parser.add_argument("--layer", type=str, default=4) #allrun
    parser.add_argument("--ratio", type=str, default='') #allrun
    parser.add_argument("--is_sim", default=True)
    parser.add_argument("--is_regress", default=False)
    args = parser.parse_args()

    if 'sicksts' in args.task or 'sts' in args.task or 'STS' in args.task:
        args.is_regress = True
        args.num_labels = 1
    if 'sim' in args.task:
        args.is_sim = True
    else:
        args.is_sim = False
    args.task_name=f'{args.task}_{args.task_name}{args.ratio}'  #allrun
    args.output_model_path = f'./models/{args.task_name}_outputmodel.bin'
    args.train_corpus = f'./datasets/{args.task}/train{args.ratio}.tsv'
    args.dev_corpus = f'./datasets/{args.task}/dev.tsv'
    args.test_corpus = f'./datasets/{args.task}/test.tsv'
    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    print('\n')
    print('seed:', args.seed)
    print('learning rate:', args.learning_rate)
    print('batch size:', args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pretrained Tokenizer
    if 'roberta' in args.pretrained_model:
        path_model = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(path_model)
        config = RobertaConfig.from_pretrained(path_model, num_labels=args.num_labels)
        config.model = 'roberta'
        config.num_layers = args.num_layers
        config.large = args.num_large
        model = RobertaForSequenceClassification.from_pretrained(path_model, config=config)
    else:
        path_model = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(path_model)
        config = BertConfig.from_pretrained(path_model, num_labels=args.num_labels)
        config.model = 'bert'
        config.num_layers = args.num_layers
        config.large = args.num_large
        model = BertForSequenceClassification.from_pretrained(path_model, config=config)
    if args.num_layers==0: #不增强
        layer_all = []
    else:
        if 'layer' in args.task_name: #在dev上测试 增强每层的效果
            layer_all = [int(args.layer)]
        else:
            layer_result = pd.read_csv(f'results/result_{args.task}_layer{args.ratio}.csv', encoding='utf-8')
            if args.is_regress:
                layer_all = layer_result['pearsonr']
            else:
                layer_all = layer_result['f1 score']
            layer_all = layer_all.to_numpy()
            args.num_layers = -int(args.num_layers)  #根据dev的测试结果，得到效果最好的前三层
            layer_all = np.argsort(layer_all)[args.num_layers:]
    
    if 'sim' in args.task:  #similarity增强
        config.sim_layer = layer_all
        config.large_layer = []
    else:  #large 增强
        config.large_layer = layer_all
        config.sim_layer = []
    print('sim layer:',config.sim_layer)
    print('large layer:',config.large_layer)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)


    # Split data into train and validation
    train_dataset = MyDataset(args, args.train_corpus)
    dev_dataset = MyDataset(args, args.dev_corpus)
    test_dataset = MyDataset(args, args.test_corpus)

    # Create train and validation dataloaders
    print('Loading the data')
    print('train datasets:', len(train_dataset))
    print('dev datasets:', len(dev_dataset))
    print('test datasets:', len(test_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    args.num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)  # 学习率预热


    with open(f"./results/result_{args.task_name}.txt", "a") as f:
        f.write('seed:{}\n'.format(args.seed))
        f.write('learning rate:{}\n'.format(args.learning_rate))
        f.write('batch size:{}\n'.format(args.batch_size))
        f.write('sim layer:{}\n'.format(config.sim_layer))  #all
        f.write('large layer:{}\n'.format(config.large_layer))  #all

    patience_counter = 0
    #best_result = 0.
    best_dev = 0.
    best_dev_result = None
    for epoch in range(args.epochs):
        model.train()  # BatchNormalization and Dropout
        time_start = time.time()
        total_loss, total_val_loss = 0, 0

        for step, batch in enumerate(train_dataloader):
            model.zero_grad()  
            input_ids, token_type_ids, attention_mask, lens = batch["input_ids"].squeeze(1), batch["token_type_ids"].squeeze(1), batch["attention_mask"].squeeze(1), \
                batch["lens"].to(device)
            if args.is_regress:
                labels = batch["labels"].unsqueeze(1).float()
            else:
                labels = batch["labels"]
            similarity = batch["similarity"].to(device)
            if not args.is_sim:
                similarity = None
            output = model(input_ids, similarity=similarity, lens=lens, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels.to(device))
            loss, logits = output[0], output[1]


            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)

            total_loss += loss.item()
            loss.backward() 
            optimizer.step()
            scheduler.step()


        avg_train_loss = total_loss / len(train_dataloader)
        avg_dev_loss, avg_dev_accuracy, dev_score = evaluate(args, model, dev_dataloader)
        avg_test_loss, avg_test_accuracy, test_score = evaluate(args, model, test_dataloader)
        if args.is_regress:
            dev_result = dev_score['pearsonr']
        else:
            dev_result = avg_dev_accuracy
        if best_dev < dev_result:
            best_dev = dev_result
            best_dev_result = [avg_dev_accuracy, dev_score]
            torch.save(model, args.output_model_path)
        with open(f"./results/result_{args.task_name}.txt", "a") as f:
            if args.is_regress:
                dev_p = dev_score['pearsonr']
                dev_sp = dev_score['spearmanr']
                test_p = test_score['pearsonr']
                test_sp = test_score['spearmanr']
                f.write(f'epoch     : {epoch}\n')
                f.write(f'Train time     : {round(time.time() - time_start, 1)}s\n')
                f.write(f'Train loss     : {avg_train_loss}\n')
                f.write(f'Validation loss: {avg_dev_loss}\n')
                f.write(f'Validation pearsonr : {dev_p}\n')
                f.write(f'Validation spearmanr : {dev_sp}\n')
                f.write(f'Test pearsonr : {test_p}\n')
                f.write(f'Test spearmanr : {test_sp}\n')
                f.write('\n')
                print(f'epoch     : {epoch}')
                print(f'Train time     : {round(time.time() - time_start, 1)}s')
                print(f'Train loss     : {avg_train_loss}')
                print(f'Validation loss: {avg_dev_loss}')
                print(f'Validation pearsonr : {dev_p}')
                print(f'Validation spearmanr : {dev_sp}')
                print(f'Test pearsonr : {test_p}')
                print(f'Test spearmanr : {test_sp}')
                print('\n')
            else:
                f.write(f'epoch     : {epoch}\n')
                f.write(f'Train time     : {round(time.time() - time_start, 1)}s\n')
                f.write(f'Train loss     : {avg_train_loss}\n')
                f.write(f'Validation loss: {avg_dev_loss}\n')
                f.write(f'Accuracy : {avg_dev_accuracy}\n')
                f.write(f'Validation F1 score : {dev_score[1][2]}\n')
                f.write(f'Test Accuracy: {avg_test_accuracy}\n')
                f.write(f'Test F1 score: {test_score[1][2]}\n')
                f.write('\n')
                print(f'epoch     : {epoch}')
                print(f'Train time     : {round(time.time() - time_start, 1)}s')
                print(f'Train loss     : {avg_train_loss}')
                print(f'Validation loss: {avg_dev_loss}')
                print(f'Accuracy : {avg_dev_accuracy}')
                print(f'Validation F1 score : {dev_score[1][2]}')
                print(f'Test Accuracy: {avg_test_accuracy}')
                print(f'Test F1 score: {test_score[1][2]}')
                print('\n')

    if 'layer' in args.task_name:
        final_score = best_dev_result[1]
        final_accuracy = best_dev_result[0]
        final_result = ['layer', args.layer]
    else:
        # Evaluation phase.
        print("Final evaluation on the test dataset.")

        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path).module.state_dict())
        else:
            model.load_state_dict(torch.load(args.output_model_path).state_dict())

        _, avg_test_accuracy, test_score = evaluate(args, model, test_dataloader)
        final_score = test_score
        final_accuracy = avg_test_accuracy
        final_result = ['seed', args.seed]
    if args.is_regress:      
        final1 = final_score['pearsonr']
        final2 = final_score['spearmanr']
        print(f'pearsonr: {final1}')
        print(f'spearmanr: {final2}')
    else:
        final1 = final_accuracy
        final2 = final_score[1][2]
        print(f'Test F1 score: {final2}')
        print(f'Test Accuracy: {final1}')

    with open(f"./results/result_{args.task_name}.txt", "a") as f:
        if args.is_regress:
            f.write("pearsonr{:.5f}, spearmanr{:.5f}\n".format(final1, final2))
        else:
            f.write('Report precision, recall, and f1:')
            f.write('\n')
            for i in range(2):
                f.write("Label {}: {:.5f}, {:.5f}, {:.5f}".format(i, final_score[i][0], final_score[i][1], final_score[i][2]))
                f.write('\n')
        f.write('\r\n')
    result_data = f"./results/result_{args.task_name}.csv" 
    if not os.path.exists(result_data):
        with open(result_data,'w', encoding='utf-8') as f:
            csv_write = csv.writer(f)
            if args.is_regress:
                csv_head = [final_result[0], "pearsonr", "spearmanr"]
            else:
                csv_head = [final_result[0], "accuary", "f1 score"]
            csv_write.writerow(csv_head)

    with open(result_data, 'a+', encoding='utf-8') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([final_result[1], final1, final2])
