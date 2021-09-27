import argparse
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score,classification_report, confusion_matrix
from pylab import rcParams
from transformers import BertTokenizerFast as BertTokenizer
from src.models.TitleDetection import TitleDetectionDataset, TitleDetection, TitleDetection_BertSequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 250921
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
MAX_TOKEN_COUNT = 128
train_df = pd.read_csv("./data/processed/train_cleaned.csv")
valid_df = pd.read_csv("./data/processed/valid_cleaned.csv")
test_df = pd.read_csv("./data/processed/test_cleaned.csv")
train_dataset = TitleDetectionDataset(
    train_df,
    tokenizer,
    max_token_len=MAX_TOKEN_COUNT
)
valid_dataset = TitleDetectionDataset(
    valid_df,
    tokenizer,
    max_token_len=MAX_TOKEN_COUNT
)
test_dataset = TitleDetectionDataset(
    test_df,
    tokenizer,
    max_token_len=MAX_TOKEN_COUNT
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)
n_other_features = len(list(train_df.columns)[1:-1])


def save_checkpoint(save_path, model, auroc):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'auroc': auroc}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['auroc']


def save_metrics(save_path, train_loss_list, valid_loss_list, auroc_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'auroc_list': auroc_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['auroc_list'], state_dict['global_steps_list']

def train(model,
          optimizer,
          train_loader=train_loader,
          valid_loader=test_loader,
          num_epochs=5,
          eval_every=len(train_loader) // 10,
          file_path="./experiment",
          best_auroc=float("-Inf"),
          patience=3
          ):

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    roc_auc_list = []
    # training loop
    not_improved = 0
    model.train()
    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                others = batch['others']
                labels = batch['labels']
                labels = labels.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                others = others.to(device)
                output = model(input_ids, attention_mask, others, labels)
                loss, _ = output
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        y_pred = []
                        y_true = []
                        #  validation loop
                        with tqdm(valid_loader, unit="batch") as vepoch:
                            for batch in vepoch:
                                input_ids = batch['input_ids']
                                attention_mask = batch['attention_mask']
                                others = batch['others']
                                labels = batch['labels']
                                labels = labels.to(device)
                                input_ids = input_ids.to(device)
                                attention_mask = attention_mask.to(device)
                                others = others.to(device)
                                output = model(
                                    input_ids, attention_mask, others, labels)
                                loss, output = output
                                valid_running_loss += loss.item()
                                y_pred.extend(output.squeeze(1).tolist())
                                y_true.extend(labels.squeeze(1).tolist())

                    # evaluation
                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)
                    roc_auc = roc_auc_score(y_true, y_pred)
                    roc_auc_list.append(roc_auc)

                    # resetting running values
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    # print progress
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, AUROC: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                                  average_train_loss, average_valid_loss, roc_auc))

                    # checkpoint
                    if best_auroc < roc_auc:
                        best_auroc = roc_auc
                        save_checkpoint(file_path + '/' + 'model.pt',
                                        model, best_auroc)
                        save_metrics(file_path + '/' + 'metrics.pt',
                                     train_loss_list, valid_loss_list, roc_auc_list, global_steps_list)
                        not_improved = 0
                    else:
                        if not_improved >= patience:
                            print(
                                "The model has not been improved for a while, early stopped")
                            return
                        else:
                            not_improved += 1

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list,
                 valid_loss_list, roc_auc_list, global_steps_list)
    print('Finished Training!')


def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    wrong_sentence = []
    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch in tepoch:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                others = batch['others']
                labels = batch['labels']
                labels = labels.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                others = others.to(device)
                output = model(input_ids, attention_mask, others, labels)
                _, output = output
                output[output > 0.5] = 1
                output[output < 0.5] = 0
                output = output.squeeze(1)
                labels = labels.squeeze(1)
                non_match = np.where(output != labels)[0]
                text = np.array(batch['text'])
                if list(non_match):
                    wrong = text[non_match]
                    features = others[non_match].tolist()
                    true_labels = labels[non_match].tolist()
                    one = list(zip(wrong, features, true_labels))
                    wrong_sentence.append(one)
                y_pred.extend(output.tolist())
                y_true.extend(labels.tolist())

    print("wrong sentences examples\n", random.sample(wrong_sentence, 5))
    roc_auc = roc_auc_score(y_true, y_pred)
    print('Classification Report:')
    print('AUROC: ', roc_auc)
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['Title', 'Non-Title'])
    ax.yaxis.set_ticklabels(['Title', 'Non-Title'])




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Title Detection')
    parser.add_argument('--bertSequence', action='store_true', default=False,
                        help='choose to use BertSequenceClassifier instead of Bert')
    parser.add_argument('--path', default="./experiment", help='the path to store the models or load the models')
    parser.add_argument('--train', action='store_true', default=False,
                        help='choose to whether train a new model')
    args = parser.parse_args()

    # training
    if args.train:
        if not args.bertSequence:
            model = TitleDetection(n_other_features).to(device)
        else:
            model = TitleDetection_BertSequence(n_other_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        train(model=model, optimizer=optimizer, file_path=args.path)

    # evaluation
    if not args.bertSequence:
        best_model = TitleDetection(n_other_features).to(device)
    else:
        best_model = TitleDetection_BertSequence(n_other_features).to(device)
    load_checkpoint(args.path + '/model.pt', best_model)
    evaluate(best_model, test_loader)


