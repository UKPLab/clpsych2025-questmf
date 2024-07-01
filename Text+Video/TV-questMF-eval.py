import pandas as pd
import scipy.io as sio
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
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
    p.add_argument("-v_ckpt", "--video_checkpoint_path", type=str, help="Path to checkpoint for the video model")
    p.add_argument("-tv_ckpt", "--tv_checkpoint_path", type=str, help="Path to checkpoint for the text+video model")
    p.add_argument("-m_files", "--missing_video_files",nargs='+', type=int, help="List of file numbers for incomplete video files")

    return (p.parse_args())

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

class dds(Dataset):
    def __init__(self,split,data_path,label_path,missing_files_list):
        super(dds,self).__init__()
        self.PAD = tokenizer_txt.pad_token_id
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
            
            vid_file = data_path + p_id + '_P/features/' + p_id + '_CNN_ResNet.mat'
            t = librosa.get_duration(path=data_path + p_id + '_P/' + p_id + '_AUDIO.wav')
            
            self.data.append([txt_list,vid_file,df_txt,t,float(phq_score_list[i])])
    
    # Text Preprocess
    def preprocess_txt(self,txt_list):
        encoded_input = tokenizer_txt(txt_list, padding=True, truncation=True, return_tensors='pt').to(device)
        l = len(txt_list)
        # Maximum number of turns
        if l > 120:
            l = 120
        # Mask for attention layer
        mask_txt = torch.tensor([False]*l)
        # Compute token embeddings
        with torch.no_grad():
            embedder_output = embedder_txt(**encoded_input)
        # Perform mean pooling
        sentence_embeddings = mean_pooling(embedder_output, encoded_input['attention_mask'])[:l]
        if l < 120:
            z = torch.zeros(120-l,768).to(device)
            sentence_embeddings = torch.cat((sentence_embeddings,z),dim=0)
            mask_txt = torch.cat((mask_txt,torch.tensor([True]*(120-l))))
        return sentence_embeddings.detach().cpu(),mask_txt

    # Video Preprocess
    def preprocess_vid(self,vid_file,df_txt,t):
        vid_data = sio.loadmat(vid_file)
        vid_feat = vid_data['feature']
        feat_len = len(vid_feat)
        
        # Sampling rate
        m_factor = feat_len/t

        # Preprocessing start and end times
        start_times = (df_txt['Start_Time'].values*m_factor).tolist()
        end_times = (df_txt['End_Time'].values*m_factor).tolist()

        x = 1
        y = len(start_times)
        while x < y:
            if start_times[x-1]>start_times[x] or start_times[x]>feat_len:
                del start_times[x]
                del end_times[x]
                x = x-1
                y = y-1
            if end_times[x-1]>end_times[x] or end_times[x]>feat_len:
                del start_times[x]
                del end_times[x]
                x = x-1
                y = y-1
            x = x+1
        
        start_times = [round(a) for a in start_times]
        end_times = [round(b) for b in end_times]
        
        vid_feat_0 = torch.tensor(vid_feat[start_times[0]:end_times[0]]).unsqueeze(0)
        # Mean pooling
        out = torch.mean(vid_feat_0,dim=1)
        out = out.detach().cpu()

        for i in range(1,len(start_times)):
            vid_feat_i = torch.tensor(vid_feat[start_times[i]:end_times[i]]).unsqueeze(0)
            # Mean pooling
            mean_i = torch.mean(vid_feat_i,dim=1)
            mean_i = mean_i.detach().cpu()
            out = torch.cat((out,mean_i),dim=0)
        l = len(start_times)
        # Maximum number of turns
        if l > 120:
            l = 120
        # Mask for attention layer
        mask_vid = torch.tensor([False]*l)
        out = out[:l]
        if l < 120:
            z = torch.zeros(120-l,2048)
            out = torch.cat((out,z),dim=0)
            mask_vid = torch.cat((mask_vid,torch.tensor([True]*(120-l))))
        return out,mask_vid
    def __getitem__(self,index):
        embedding_txt, mask_txt = self.preprocess_txt(self.data[index][0])
        embedding_vid, mask_vid = self.preprocess_vid(self.data[index][1],self.data[index][2],self.data[index][3])
        return [embedding_txt,mask_txt,embedding_vid,mask_vid,self.data[index][4]]
    def __len__(self):
        return len(self.data)

class lstm_regressor_txt(nn.Module):
    def __init__(self):
        super(lstm_regressor_txt, self).__init__()
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

class lstm_regressor_vid(nn.Module):
    def __init__(self):
        super(lstm_regressor_vid, self).__init__()
        self.lstm_1 = nn.LSTM(2048,50,batch_first=True,bidirectional=True)
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

class lstm_regressor(nn.Module):
    def __init__(self,txt_model,vid_model):
        super(lstm_regressor, self).__init__()
        self.txt_model = txt_model
        self.vid_model = vid_model

        self.cross_vid_txt = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.cross_txt_vid = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)

        self.self_vid_txt = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.self_txt_vid = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)

        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.8),
                                nn.Linear(24000,256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,1))

        # Freeze Text Encoder
        for param_txt in self.txt_model.parameters():
            param_txt.requires_grad = False

    def forward(self,C_txt,key_padding_mask_txt,C_vid,key_padding_mask_vid):

        _,c_att_txt = self.txt_model(C_txt,key_padding_mask_txt)

        _,c_att_vid = self.vid_model(C_vid,key_padding_mask_vid)

        c_vid_txt,_ = self.cross_vid_txt(c_att_vid,c_att_txt,c_att_txt,key_padding_mask=key_padding_mask_txt)
        c_txt_vid,_ = self.cross_txt_vid(c_att_txt,c_att_vid,c_att_vid,key_padding_mask=key_padding_mask_vid)

        c_att_vid_txt,_ = self.self_vid_txt(c_vid_txt,c_vid_txt,c_vid_txt,key_padding_mask=key_padding_mask_txt)
        c_att_txt_vid,_ = self.self_txt_vid(c_txt_vid,c_txt_vid,c_txt_vid,key_padding_mask=key_padding_mask_vid)

        c_comb = torch.cat((c_att_vid_txt,c_att_txt_vid),dim=2)

        pred = self.mlp(c_comb)

        return pred

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
        c_txt, mask_txt, c_vid, mask_vid, phq_scores = tuple(t.to(device) for t in batch)

        # Compute predictions
        with torch.no_grad():
            preds1 = model1.forward(c_txt,mask_txt,c_vid,mask_vid)
            preds2 = model2.forward(c_txt,mask_txt,c_vid,mask_vid)
            preds3 = model3.forward(c_txt,mask_txt,c_vid,mask_vid)
            preds4 = model4.forward(c_txt,mask_txt,c_vid,mask_vid)
            preds5 = model5.forward(c_txt,mask_txt,c_vid,mask_vid)
            preds6 = model6.forward(c_txt,mask_txt,c_vid,mask_vid)
            preds7 = model7.forward(c_txt,mask_txt,c_vid,mask_vid)
            preds8 = model8.forward(c_txt,mask_txt,c_vid,mask_vid)

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
    
    # val_accuracy = np.mean(val_accuracy)

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
    tokenizer_txt = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    embedder_txt = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
    
    # Datasets
    data_train = dds('train',args.data_path,args.label_path,args.missing_video_files)
    data_val = dds('val',args.data_path,args.label_path,args.missing_video_files)
    data_test = dds('test',args.data_path,args.label_path,args.missing_video_files)
    
    # Define Text Encoder for each Question
    t1 = lstm_regressor_txt()
    t2 = lstm_regressor_txt()
    t3 = lstm_regressor_txt()
    t4 = lstm_regressor_txt()
    t5 = lstm_regressor_txt()
    t6 = lstm_regressor_txt()
    t7 = lstm_regressor_txt()
    t8 = lstm_regressor_txt()
    
    # Load pretrained weights for text encoder
    t1.load_state_dict(torch.load(args.text_checkpoint_path+'-phq1-parameters.pt'))
    t2.load_state_dict(torch.load(args.text_checkpoint_path+'-phq2-parameters.pt'))
    t3.load_state_dict(torch.load(args.text_checkpoint_path+'-phq3-parameters.pt'))
    t4.load_state_dict(torch.load(args.text_checkpoint_path+'-phq4-parameters.pt'))
    t5.load_state_dict(torch.load(args.text_checkpoint_path+'-phq5-parameters.pt'))
    t6.load_state_dict(torch.load(args.text_checkpoint_path+'-phq6-parameters.pt'))
    t7.load_state_dict(torch.load(args.text_checkpoint_path+'-phq7-parameters.pt'))
    t8.load_state_dict(torch.load(args.text_checkpoint_path+'-phq8-parameters.pt'))
    
    t1.to(device)
    t2.to(device)
    t3.to(device)
    t4.to(device)
    t5.to(device)
    t6.to(device)
    t7.to(device)
    t8.to(device)

    # Define Video Encoder for each Question
    v1 = lstm_regressor_vid()
    v2 = lstm_regressor_vid()
    v3 = lstm_regressor_vid()
    v4 = lstm_regressor_vid()
    v5 = lstm_regressor_vid()
    v6 = lstm_regressor_vid()
    v7 = lstm_regressor_vid()
    v8 = lstm_regressor_vid()
    
    # Load pretrained weights for video encoder
    v1.load_state_dict(torch.load(args.video_checkpoint_path+'-phq1-sr-parameters.pt'))
    v2.load_state_dict(torch.load(args.video_checkpoint_path+'-phq2-sr-parameters.pt'))
    v3.load_state_dict(torch.load(args.video_checkpoint_path+'-phq3-sr-parameters.pt'))
    v4.load_state_dict(torch.load(args.video_checkpoint_path+'-phq4-sr-parameters.pt'))
    v5.load_state_dict(torch.load(args.video_checkpoint_path+'-phq5-sr-parameters.pt'))
    v6.load_state_dict(torch.load(args.video_checkpoint_path+'-phq6-sr-parameters.pt'))
    v7.load_state_dict(torch.load(args.video_checkpoint_path+'-phq7-sr-parameters.pt'))
    v8.load_state_dict(torch.load(args.video_checkpoint_path+'-phq8-sr-parameters.pt'))
    
    v1.to(device)
    v2.to(device)
    v3.to(device)
    v4.to(device)
    v5.to(device)
    v6.to(device)
    v7.to(device)
    v8.to(device)

    # Define T+V fusion models for each Question
    m1 = lstm_regressor(t1,v1)
    m2 = lstm_regressor(t2,v2)
    m3 = lstm_regressor(t3,v3)
    m4 = lstm_regressor(t4,v4)
    m5 = lstm_regressor(t5,v5)
    m6 = lstm_regressor(t6,v6)
    m7 = lstm_regressor(t7,v7)
    m8 = lstm_regressor(t8,v8)
    
    # Load pretrained weights for T+V fusion models
    m1.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq1-sr-parameters.pt'))
    m2.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq2-sr-parameters.pt'))
    m3.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq3-sr-parameters.pt'))
    m4.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq4-sr-parameters.pt'))
    m5.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq5-sr-parameters.pt'))
    m6.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq6-sr-parameters.pt'))
    m7.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq7-sr-parameters.pt'))
    m8.load_state_dict(torch.load(args.tv_checkpoint_path+'-phq8-sr-parameters.pt'))
    
    m1.to(device)
    m2.to(device)
    m3.to(device)
    m4.to(device)
    m5.to(device)
    m6.to(device)
    m7.to(device)
    m8.to(device)

    # Dataloaders
    train_dataloader = DataLoader(data_train,  batch_size=10)
    val_dataloader = DataLoader(data_val,  batch_size=10)
    test_dataloader = DataLoader(data_test,  batch_size=10)
    
    # Specify loss functions/Metrics
    ccc_loss_fn = ccc_loss()
    loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()
    
    # Evaluate trained T+V model
    print(evaluate(m1,m2,m3,m4,m5,m6,m7,m8,test_dataloader))