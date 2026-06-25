import pandas as pd
import pickle
import torch
import tqdm
import argparse
import os
from models import adSent_Mistral,adSent_llama,adSent_qwen

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='meta-llama/Llama-3.1-8B-Instruct') #{Open-Orca/Mistral-7B-OpenOrca,gpt-3.5-turbo,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-72B-Instruct}
    parser.add_argument("--prompt", type=str, default="Rewrite the following article with {} sentiment but do not change any facts. please also do not summarize or expand or inlcude pormpt in response. try to be concise and only change the sentiment")
    parser.add_argument("--dataset", type=str, default="politifact") #{politifact, gossipcop, lun}
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--max_seq_length", type=int,default=1024)
    parser.add_argument("--news_value", default='sen') #sentiment
    parser.add_argument("--sentiment",type=str,choices=["positive", "negative", "neutral"], default="neutral")
    
    args = parser.parse_args()
    print(args)
    return args


def get_prompt(prompt,news_txt):
    
    p=f"{prompt}:{news_txt}\nAnswer:"
    
    return p

def main():
    args = get_args()
    ####### Read_data #############################################################################
   
    file_path = os.path.join(args.base_dir, f"{args.dataset}_test.pkl")
    with open(file_path, "rb") as file:
        data_test = pickle.load(file)
    ################ Load the model #################################################################
    #model=adSent_qwen(model=args.model_name,tokenizer=args.model_name,max_length=args.max_seq_length)
    model=adSent_llama(model=args.model_name,tokenizer=args.model_name,max_length=args.max_seq_length)
    ############### input data to the model ###########################################################
    data_test['adsent']=[]
    for i in tqdm.tqdm(range(len(data_test["news"]))):
        news_text = data_test["news"][i]
        formatted_prompt = args.prompt.format(args.sentiment)
        prompt=get_prompt(formatted_prompt,news_text)
        full_response,filtered_response=model.sent_attack(prompt)
        data_test['adsent'].append(filtered_response)
        data_test['full_output']=full_response
    
################### save the new dataset #######################################################
    del data_test["news"]
    data_test['news'] = data_test.pop('adsent')
    data_test['prompt']=args.prompt
    data_test['model_name']=args.model_name
    sent_tag = args.sentiment[:3]
    with open(os.path.join(args.base_dir,f"{args.dataset}_test_{args.news_value}_llama_{sent_tag}.pkl"), 'wb') as f:
        pickle.dump(data_test, f) 
if __name__ == "__main__":
    main()