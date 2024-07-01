from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import argparse
import os
from transformers import get_linear_schedule_with_warmup

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-d_path", "--data_path", type=str, help="Path to data files (text transcripts, audio files, video features)")
    p.add_argument("-l_path", "--label_path", type=str, help="Path to labels, i.e., PHQ-8 scores")
    p.add_argument("-t_ckpt", "--text_checkpoint_path", type=str, help="Path to checkpoint for the text model")
    p.add_argument("-m_files", "--missing_video_files",nargs='+', type=int, help="List of file numbers for incomplete video files")

    return (p.parse_args())

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class dds(Dataset):
    def __init__(self,split,data_path,label_path,missing_files_list):
        super(dds,self).__init__()
        if split == 'train':
            df_data = pd.read_csv(label_path + 'train_split.csv')
        elif split == 'val':
            df_data = pd.read_csv(label_path + 'dev_split.csv')
        elif split == 'test':
            df_data = pd.read_csv(label_path + 'test_split.csv')
        else:
            raise Exception(f"wrong split: {split}")
        self.data = []
        p_id_list = df_data['Participant_ID'].tolist()
        phq_score_list = df_data['PHQ_Score'].tolist()
        for i in range(len(p_id_list)):
            p_id = str(p_id_list[i])
            # Files not complete for video
            if p_id_list[i] in missing_files_list:
                continue
            df_txt = pd.read_csv(data_path + p_id + '_P/' + p_id + '_Transcript.csv')
            txt_list = df_txt['Text'].tolist()
            self.data.append([txt_list,float(phq_score_list[i])])
    
    def preprocess(self,txt_list):
        encoded_input = tokenizer(txt_list, padding=True, truncation=True, return_tensors='pt').to(device)
        l = len(txt_list)
        # Maximum number of turns
        if l > 120:
            l = 120
        # Mask for attention layer
        mask = torch.tensor([False]*l)
        # Compute token embeddings
        with torch.no_grad():
            embedder_output = embedder(**encoded_input)
        # Perform mean pooling
        sentence_embeddings = mean_pooling(embedder_output, encoded_input['attention_mask'])[:l]
        if l < 120:
            z = torch.zeros(120-l,768).to(device)
            sentence_embeddings = torch.cat((sentence_embeddings,z),dim=0)
            mask = torch.cat((mask,torch.tensor([True]*(120-l))))
        return sentence_embeddings.detach().cpu(),mask
    def __getitem__(self,index):
        embedding, mask = self.preprocess(self.data[index][0])
        return [embedding,mask,self.data[index][1]]
    def __len__(self):
        return len(self.data)

class lstm_regressor(nn.Module):
    def __init__(self):
        super(lstm_regressor, self).__init__()
        self.lstm_1 = nn.LSTM(768,50,batch_first=True,bidirectional=True)
        self.attention = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.5)
        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.5),
                                nn.Linear(12000,256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,1))

    def forward(self,C,key_padding_mask):
        c_lstm,_ = self.lstm_1(C)
        c_att,_ = self.attention(c_lstm,c_lstm,c_lstm,key_padding_mask=key_padding_mask)
        pred = self.mlp(c_att)
        return pred,c_att

# CCC loss
class ccc_loss(nn.Module):
    def __init__(self):
        super(ccc_loss, self).__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std
    def forward(self, prediction, ground_truth):
        mean_gt = self.mean (ground_truth, 0)
        mean_pred = self.mean (prediction, 0)
        var_gt = self.var (ground_truth, 0)
        var_pred = self.var (prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum (v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2)))
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator=2*cor*sd_gt*sd_pred
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        ccc = numerator/denominator
        return 1-ccc

def evaluate(model1,model2,model3,model4,model5,model6,model7,model8, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the models into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()
    model8.eval()

    # Tracking variables
    preds_list = []
    labels_list = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        c, mask, phq_scores = tuple(t.to(device) for t in batch)

        # Compute predictions
        with torch.no_grad():
            preds1,_ = model1.forward(c,mask)
            preds2,_ = model2.forward(c,mask)
            preds3,_ = model3.forward(c,mask)
            preds4,_ = model4.forward(c,mask)
            preds5,_ = model5.forward(c,mask)
            preds6,_ = model6.forward(c,mask)
            preds7,_ = model7.forward(c,mask)
            preds8,_ = model8.forward(c,mask)

        # Get total score
        preds = preds1.squeeze(1)+preds2.squeeze(1)+preds3.squeeze(1)+preds4.squeeze(1)+preds5.squeeze(1)+preds6.squeeze(1)+preds7.squeeze(1)+preds8.squeeze(1)
        preds_list.append(preds)
        labels_list.append(phq_scores)

    # Compute the CCC, RMSE and MAE
    preds_all = torch.cat(preds_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    val_loss_ccc = 1 - ccc_loss_fn(preds_all, labels_all.float())
    val_loss_rmse = torch.sqrt(loss_fn(preds_all, labels_all.float()))
    val_loss_mae = mae_loss_fn(preds_all,labels_all.float())

    return val_loss_ccc,val_loss_rmse,val_loss_mae

if __name__ == '__main__':
    set_seed(42)    # Set seed for reproducibility

    args = cmdline_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"# Using device: {device}")

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    embedder = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
    
    # Datasets
    data_train = dds('train', args.data_path,args.label_path,args.missing_video_files)
    data_val = dds('val', args.data_path,args.label_path,args.missing_video_files)
    data_test = dds('test', args.data_path,args.label_path,args.missing_video_files)
    
    # Define Text Encoder for each Question
    r1 = lstm_regressor()
    r2 = lstm_regressor()
    r3 = lstm_regressor()
    r4 = lstm_regressor()
    r5 = lstm_regressor()
    r6 = lstm_regressor()
    r7 = lstm_regressor()
    r8 = lstm_regressor()
    
    # Load pretrained weights
    r1.load_state_dict(torch.load(args.text_checkpoint_path + '-phq1-parameters.pt'))
    r2.load_state_dict(torch.load(args.text_checkpoint_path + '-phq2-parameters.pt'))
    r3.load_state_dict(torch.load(args.text_checkpoint_path + '-phq3-parameters.pt'))
    r4.load_state_dict(torch.load(args.text_checkpoint_path + '-phq4-parameters.pt'))
    r5.load_state_dict(torch.load(args.text_checkpoint_path + '-phq5-parameters.pt'))
    r6.load_state_dict(torch.load(args.text_checkpoint_path + '-phq6-parameters.pt'))
    r7.load_state_dict(torch.load(args.text_checkpoint_path + '-phq7-parameters.pt'))
    r8.load_state_dict(torch.load(args.text_checkpoint_path + '-phq8-parameters.pt'))
    
    r1.to(device)
    r2.to(device)
    r3.to(device)
    r4.to(device)
    r5.to(device)
    r6.to(device)
    r7.to(device)
    r8.to(device)
    
    # Dataloaders
    train_dataloader = DataLoader(data_train,  batch_size=10)
    val_dataloader = DataLoader(data_val,  batch_size=10)
    test_dataloader = DataLoader(data_test,  batch_size=10)

    # Specify loss functions/Metrics
    ccc_loss_fn = ccc_loss()
    loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()
    
    # Evaluate trained T model
    print(evaluate(r1,r2,r3,r4,r5,r6,r7,r8,test_dataloader))