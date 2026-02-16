import torch
import os
p = 'models_saved/la_nn_best.pth'
print('exists', os.path.exists(p))
ckp = torch.load(p, map_location='cpu')
print('type:', type(ckp))
if isinstance(ckp, dict):
    print('top-level keys:', list(ckp.keys()))
    sd = ckp.get('state_dict') or ckp.get('model_state_dict') or ckp
else:
    sd = ckp
print('\n--- state_dict sample keys (first 120) ---')
for i, k in enumerate(list(sd.keys())[:120]):
    print(i, k)
print('\ntotal keys =', len(sd))
