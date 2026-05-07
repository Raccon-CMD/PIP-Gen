import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ESM model and tokenizer (ensure path is correct)
tokenizer_name = r'D:\python code\PIP-Gen\prediction\esm2_t12_35M_UR50D'
model_name = r'D:\python code\PIP-Gen\prediction\esm2_t12_35M_UR50D'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
model = AutoModel.from_pretrained(model_name, config=config).to(device)


def extract_token_features(sequence, tokenizer, model, device):
    """
    Returns a token-level feature matrix (L x D) for the input protein sequence.
    L = sequence length (including special tokens), D = hidden size.
    """
    inputs = tokenizer(sequence, return_tensors='pt')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # tuple of (num_layers+1) * [1, L, D]
    # Average the last 4 hidden states across layers, then squeeze
    feature_matrix = torch.stack(hidden_states[-4:]).mean(0).squeeze(0).cpu().numpy()
    return feature_matrix


# ==================== Load sequence data ====================
df = pd.read_csv(r'D:\python code\PIP-Gen\data\PIP Data\PIP_Test.csv', index_col=0)
df = df.sample(frac=1, random_state=0)  # shuffle

# ==================== Extract and pool features ====================
pooled_features = []

output_csv_path = r'D:\python code\PIP-Gen\PIP_Train_ESM_features.csv'

for idx, row in df.iterrows():
    sequence = row['aa_seq']  # assuming the column name is 'aa_seq'
    token_feats = extract_token_features(sequence, tokenizer, model, device)  # shape (L, D)

    # Average over sequence length to obtain a fixed-length global feature vector (D,)
    pooled = np.mean(token_feats, axis=0)
    pooled_features.append(pooled)

if pooled_features:
    features_arr = np.array(pooled_features)
    D = features_arr.shape[1]
    col_names = [f'Feature_{i + 1}' for i in range(D)]
    result_df = pd.DataFrame(features_arr, columns=col_names)
    # Optionally include other metadata (e.g., label column) from original df
    # result_df['label'] = df['label'].values
    result_df.to_csv(output_csv_path, index=False)
    print(f'Feature extraction completed, saved to: {output_csv_path}')
    print(f'Number of samples: {len(result_df)}, feature dimension: {D}')
else:
    print('No features extracted, please check input data')