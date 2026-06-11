import pandas as pd
import pickle
import torch
import tqdm
import argparse
import os
from models import adSent_Mistral,adSent_llama,adSent_qwen
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
from datetime import datetime
import json
import re
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss


def setup_with_args(args,outdir,run_desc=""):
    if  run_desc !=None and args.desc!=None:
        run_desc+="-"+args.desc
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    args.cur_run_id=cur_run_id
    assert not os.path.exists(args.run_dir)
    # Create output directory.
    print(f'Creating output directory...{args.run_dir}')
    os.makedirs(args.run_dir)
    return args.run_dir,args

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["inference", "train"], default="inference")
    parser.add_argument("--use_finetuned_model", action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--model_type", type=str, choices=["llama", "qwen", "mistral"], default="llama")
    parser.add_argument("--model_name", default='meta-llama/Llama-3.1-8B-Instruct') #{Open-Orca/Mistral-7B-OpenOrca, meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,pretrained_model}
    parser.add_argument("--checkpoint_dir", default='') #./outputs/llama-31-8b-instruct/00024-2025-07-25_14-30-54
    #parser.add_argument("--prompt", type=str, default="You are given a news article. Your specific task is to evaluate its factual correctness. Is this article factually accurate? Answer with one word only: fake or real")
    parser.add_argument("--prompt", type=str, default="Is this news article fake or real? answer with only one word, fake or real")
    #parser.add_argument("--testset", type=str, default="gossipcop_test_sen_llama_neutral") #{'politifact_test_adv_D',''}
    parser.add_argument("--testset", type=str, default="gossipcop_test") #{'politifact_test_adv_D',''}
    parser.add_argument("--trainset", type=str, default="politifact_train_sen_llama_neutral")#{politifact_train_sen_llama_neutral}
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--max_seq_length", type=int,default=1024)
    parser.add_argument("--pbc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--desc", type=str)
    parser.add_argument("--epochs", type=int, default=1)


    args = parser.parse_args()
    print(args)
    return args

def get_prompt(prompt,news_txt):
    p=f"{prompt}:{news_txt}\nAnswer:"
    return p

def load_model(args, model_path=None):
    path = model_path if model_path is not None else args.model_name

    if args.model_type == "llama":
        return adSent_llama(model=path, tokenizer=path, max_length=args.max_seq_length)
    elif args.model_type == "qwen":
        return adSent_qwen(model=path, tokenizer=path, max_length=args.max_seq_length)
    elif args.model_type == "mistral":
        return adSent_Mistral(model=path, tokenizer=path, max_length=args.max_seq_length)

class PromptDataset(Dataset):
    def __init__(self, x, y, prompt_template):
        self.x = x
        self.y = y
        self.prompt_template = prompt_template
        #self.label_map = {0: "real", 1: "fake"}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        news = self.x[idx]
        #label_word = self.label_map[self.y[idx]]
        label_word = self.y[idx]
        prompt = f"{self.prompt_template}:{news}\nAnswer:"
        #prompt = f"{self.prompt_template}:{news}\nAnswer: {label_word}"
        return prompt, label_word

def inference(args):
    ####### Read_data #############################################################################
    file_path = os.path.join(args.base_dir, f"{args.testset}.pkl")
    with open(file_path, "rb") as file:
        data_test = pickle.load(file)
    data_test['prediction']=[]
    if args.use_finetuned_model==False:
        ################ Load the model #################################################################
        model = load_model(args)
        ############### input data to the model ###########################################################
        for i in tqdm.tqdm(range(len(data_test["news"]))):
            news_text = data_test["news"][i]
            prompt=get_prompt(args.prompt,news_text)
            if args.pbc==True:
                response,_=model.get_response_classification_pbc(prompt)
            else:
                response=model.get_response_classification(prompt)
            #print(response)
            if response=='fake':
                data_test['prediction'].append(1)
            elif response=='real':
                data_test['prediction'].append(0)
            else:
                data_test['prediction'].append(-1)
    else:
            model=adSent_llama(model=args.checkpoint_dir,tokenizer=args.checkpoint_dir,max_length=args.max_seq_length)
            model._model.eval()
            for i in tqdm.tqdm(range(len(data_test["news"]))):
                news_text = data_test["news"][i]
                prompt=get_prompt(args.prompt,news_text)
                label_logits = model.training_classification_style_inference(prompt)
                pred = torch.argmax(label_logits).item()
                print(pred) 
                data_test['prediction'].append(pred)
    with open(f'{args.run_dir}/{args.testset}_predictions.pkl', 'wb') as f:
        pickle.dump(data_test, f)
    
################################################ Metrics #######################################################
    acc = accuracy_score(data_test['labels'], data_test['prediction'])
    # Compute Macro scores
    precision_macro, recall_macro, fscore_macro, _= score(data_test['labels'], data_test['prediction'], average='macro')

    # Compute per-class scores
    precision_per_class, recall_per_class, fscore_per_class, support = score(data_test['labels'], data_test['prediction'])
    
    classes = unique_labels(data_test['labels'], data_test['prediction'])
    results = {
    'accuracy': [acc],
    'precision_macro': [precision_macro],
    'recall_macro': [recall_macro],
    'f1_macro': [fscore_macro],}
    
    # Add per-class scores
    for idx, class_name in enumerate(classes):
        results[f'precision_{class_name}'] = [precision_per_class[idx]]
        results[f'recall_{class_name}'] = [recall_per_class[idx]]
        results[f'f1_{class_name}'] = [fscore_per_class[idx]]
        results[f'support_{class_name}'] = [support[idx]]
    
    
    results_df = pd.DataFrame(results)
    
    print("Total_Test_Accuracy: {:.4f} | Prec_Macro: {:.4f} | Rec_Macro: {:.4f} | F1_Macro: {:.4f}".format(
        acc, precision_macro, recall_macro, fscore_macro))

    results_df.to_csv(f'{args.run_dir}/{args.testset}_metrics.csv', index=False)
    with open(os.path.join(args.run_dir,'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def train(args):
    ####### Read_data #############################################################################
    file_path_train = os.path.join(args.base_dir, f"{args.trainset}.pkl")
    with open(file_path_train, "rb") as file:
        train_dict= pickle.load(file)

    with open(os.path.join(args.base_dir, "politifact_train.pkl"), "rb") as file:
        train_dict_original= pickle.load(file)
    keys=['news','labels']
    train_dict = {key: train_dict[key] + train_dict_original[key] for key in keys}
    
    file_path_test = os.path.join(args.base_dir, f"{args.testset}.pkl")
    with open(file_path_test, "rb") as file:
        test_dict = pickle.load(file)
    
    x_train, y_train = train_dict['news'], train_dict['labels']
    label_map = {0: "real", 1: "fake"}
    y_train = [label_map[label] for label in train_dict['labels']]
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    # Prepare dataset and dataloader
    dataset = PromptDataset(x_train, y_train, args.prompt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    ###################### Load the model ###############################################################
    model = adSent_llama(model=args.model_name, tokenizer=args.model_name, max_length=args.max_seq_length)
    model._model.train()
    #loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model._model.parameters(), lr=2e-5)
    num_training_steps = args.epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps)
    
     ##################### Training loop #################################################################
    for epoch in range(args.epochs):
        total_loss = 0
        total_preds=0
        correct_preds=0
        for prompt, label in tqdm.tqdm(dataloader):    
            #print(f'This is prompt:{prompt[0]}')
            #print(f'This is label:{label[0]}')          
            #loss = model.training(prompt[0], label[0])
            loss,label_logits = model.training_classification_style(prompt[0], label[0])
            #print(f'This is loss:{loss}') 
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            pred = torch.argmax(label_logits).item()  # 0 or 1
            #print(f'This is prediction:{pred}')
            GT = 0 if label[0] == "real" else 1
            if pred == GT:
                correct_preds += 1
            total_preds += 1
           
        train_acc = correct_preds / total_preds
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss:.4f} - Train Accuracy: {train_acc:.4f}")
    model._model.save_pretrained(args.run_dir)
    model._tokenizer.save_pretrained(args.run_dir)
    ######################## Test on Test data ########################################################        
    #model = adSent_llama(model=args.finetuned_model_path, tokenizer=args.finetuned_model_path, max_length=args.max_seq_length)
    model._model.eval()
    test_dict['prediction']=[]
    for i in tqdm.tqdm(range(len(test_dict["news"]))):
        news_text = test_dict["news"][i]
        prompt=get_prompt(args.prompt,news_text)
        label_logits = model.training_classification_style_inference(prompt)
        pred = torch.argmax(label_logits).item() 
        test_dict['prediction'].append(pred)

    with open(f'{args.run_dir}/{args.testset}_predictions.pkl', 'wb') as f:
        pickle.dump(test_dict, f)
    
################################################ Metrics #######################################################
    acc = accuracy_score(test_dict['labels'], test_dict['prediction'])
    # Compute Macro scores
    precision_macro, recall_macro, fscore_macro, _= score(test_dict['labels'], test_dict['prediction'], average='macro')

    # Compute per-class scores
    precision_per_class, recall_per_class, fscore_per_class, support = score(test_dict['labels'], test_dict['prediction'])
    
    classes = unique_labels(test_dict['labels'], test_dict['prediction'])
    results = {
    'accuracy': [acc],
    'precision_macro': [precision_macro],
    'recall_macro': [recall_macro],
    'f1_macro': [fscore_macro],}
    
    # Add per-class scores
    for idx, class_name in enumerate(classes):
        results[f'precision_{class_name}'] = [precision_per_class[idx]]
        results[f'recall_{class_name}'] = [recall_per_class[idx]]
        results[f'f1_{class_name}'] = [fscore_per_class[idx]]
        results[f'support_{class_name}'] = [support[idx]]
    
    results_df = pd.DataFrame(results)
    
    print("Total_Test_Accuracy: {:.4f} | Prec_Macro: {:.4f} | Rec_Macro: {:.4f} | F1_Macro: {:.4f}".format(
        acc, precision_macro, recall_macro, fscore_macro))

    results_df.to_csv(f'{args.run_dir}/{args.testset}_metrics.csv', index=False)
    with open(os.path.join(args.run_dir,'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def main():
    args = get_args()
    ########### create output folder ######################################################################
    model_name=args.model_name.split('/')[-1]
    dir_name=model_name.replace(".", "").lower()
    _,args=setup_with_args(args,f'./outputs/{dir_name}','{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    ################ main loop ####################
    if args.mode == "inference":
        inference(args)
    elif args.mode == "train":
        train(args)
    


if __name__ == "__main__":
    main()