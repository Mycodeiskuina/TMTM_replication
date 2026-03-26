import torch
from tqdm import tqdm
from transformers import pipeline
import os
import pandas as pd
import json
import gc # Para limpiar la memoria RAM

user = pd.read_json('Dataset/user.json')
user_text = list(user['description'])

# Prepare the user index
user_idx = user['id']
uid_index = {uid: index for index, uid in enumerate(user_idx.values)}

# 1. OPTIMIZACIÓN DE CARGA DE TWEETS
print("Extracting each user's tweets...")
id_tweet = {i: [] for i in range(len(user_idx))}

for i in range(9):
    name = f'tweet_{i}.json'
    filepath = os.path.join("Dataset", name)
    
    if not os.path.exists(filepath):
        continue
        
    print(f"\nCargando archivo: {name}...")
    with open(filepath, 'r') as f:
        user_tweets = json.load(f)
    
    for each in tqdm(user_tweets, desc=f"Procesando {name}"):
        uid = 'u' + str(each['author_id'])
        if uid in uid_index:
            index = uid_index[uid]
            # Solo guardamos hasta 20 tweets por usuario, ¡ahorra gigabytes de RAM!
            if len(id_tweet[index]) < 20: 
                id_tweet[index].append(each['text'])
                
    # Liberamos la memoria del archivo recién procesado
    del user_tweets
    gc.collect()

folder_path = 'Dataset/processed_data'
processed_data_path = os.path.join(folder_path, "") 
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Ya no guardamos ni cargamos el JSON redundante
each_user_tweets = id_tweet 

# Load the model RoBERTa
print("\nCargando modelo RoBERTa...")
feature_extract = pipeline(
    'feature-extraction', 
    model='roberta-base', 
    tokenizer='roberta-base', 
    device=0, # Usa la A100
    padding=True, 
    truncation=True, 
    max_length=50, 
    add_special_tokens=True
)

def Des_embbeding():
    print('Running description embedding...')
    path = processed_data_path + "des_tensor.pt"
    des_vec = []
    
    for each in tqdm(user_text):
        if not each: # Maneja None y strings vacíos
            des_vec.append(torch.zeros(768))
        else:
            feature = torch.tensor(feature_extract(each)) # Tamaño: [1, seq_len, 768]
            # 2. OPTIMIZACIÓN MATEMÁTICA: Hacemos el promedio de las palabras en 1 línea
            feature_tensor = feature.squeeze(0).mean(dim=0)
            des_vec.append(feature_tensor.cpu().detach())
            
    des_tensor = torch.stack(des_vec, 0)
    torch.save(des_tensor, path)
    print('Finished descriptions.')
    return des_tensor

def tweets_embedding():
    print('Running tweets embedding...')
    path = processed_data_path + "tweets_tensor.pt"
    tweets_list = []
    
    for i in tqdm(range(len(each_user_tweets))):
        tweets = each_user_tweets[i] # Como ya no cargamos desde JSON, las llaves son enteros, no str(i)
        
        if not tweets:
            tweets_list.append(torch.zeros(768))
            continue
            
        user_tweet_tensors = []
        for tweet in tweets: # Máximo iterará 20 veces gracias a la optimización #1
            if tweet:
                feature = torch.tensor(feature_extract(tweet))
                tweet_tensor = feature.squeeze(0).mean(dim=0)
                user_tweet_tensors.append(tweet_tensor)
                
        if not user_tweet_tensors:
            tweets_list.append(torch.zeros(768))
        else:
            # 3. OPTIMIZACIÓN MATEMÁTICA: Promediamos los 20 tweets del usuario de golpe
            total_each_person_tweets = torch.stack(user_tweet_tensors).mean(dim=0)
            tweets_list.append(total_each_person_tweets.cpu().detach())
            
    tweet_tensor = torch.stack(tweets_list)
    torch.save(tweet_tensor, path)
    print('Finished tweets.')

Des_embbeding()
tweets_embedding()
