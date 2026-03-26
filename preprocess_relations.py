import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime as dt
import json
import os

# --- Configuración de Rutas ---
base_path = './Dataset' 
folder_path = './Dataset/processed_data'

print('loading raw data')
with open(os.path.join(folder_path, 'uid_index.json'), 'r') as f:
    uid_index = json.load(f)

edge = pd.read_csv(os.path.join(base_path, "edge.csv"))

print('extracting edge_index&edge_type')
edge_index = []
edge_type = []

# Primer bucle original (Followers y Following)
for i in tqdm(range(len(edge))):
    sid = str(edge['source_id'][i]) # Convertir a str para coincidir con el JSON
    tid = str(edge['target_id'][i])
    if edge['relation'][i] == 'followers':
        try:
            edge_index.append([uid_index[sid], uid_index[tid]])
            edge_type.append(0)
        except KeyError:
            continue
    elif edge['relation'][i] == 'following':
        try:
            edge_index.append([uid_index[sid], uid_index[tid]])
            edge_type.append(1)
        except KeyError:
            continue

list_membership = {}

# Segundo bucle original (Repite followers/following y prepara listas)
for i in range(len(edge)):
    sid = str(edge['source_id'][i])
    tid = str(edge['target_id'][i])
    relation = edge['relation'][i]
    
    if relation == 'followers':
        try:
            edge_index.append([uid_index[sid], uid_index[tid]])
            edge_type.append(0)
        except KeyError:
            continue
    elif relation == 'following':
        try:
            edge_index.append([uid_index[sid], uid_index[tid]])
            edge_type.append(1)
        except KeyError:
            continue
    elif relation == 'own':
        if tid not in list_membership:
            list_membership[tid] = {'creator': [], 'members': []}
        if sid in uid_index: # Verificación para evitar KeyError
            list_membership[tid]['creator'].append(uid_index[sid])
    elif relation in ['followed', 'membership']:
        if sid not in list_membership:
            list_membership[sid] = {'creator': [], 'members': []}
        if tid in uid_index: # Verificación para evitar KeyError
            list_membership[sid]['members'].append(uid_index[tid])       

# Create relation 2: Ownership
for list_id, roles in list_membership.items():
    # Conexiones del creador a cada miembro
    for creator in roles['creator']:
        for member in roles['members']:
            try:
                edge_index.append([creator, member])  # Creador a miembro
                edge_type.append(2)  # Ownership
            except KeyError:
                continue

# Guardar en la misma carpeta que tus otros tensores
torch.save(torch.LongTensor(edge_index).t(), os.path.join(folder_path, "edge_index.pt"))
torch.save(torch.LongTensor(edge_type), os.path.join(folder_path, "edge_type.pt"))
