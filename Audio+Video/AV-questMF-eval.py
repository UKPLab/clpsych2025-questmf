import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import librosa
import random
import argparse
import os

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--seed", type=int, help="Set a random seed")
    p.add_argument("-d_path", "--data_path", type=str, help="Path to data files (text transcripts, audio files, video features)")
    p.add_argument("-l_path", "--label_path", type=str, help="Path to labels, i.e., PHQ-8 scores")
    p.add_argument("-a_ckpt", "--audio_checkpoint_path", type=str, help="Path to checkpoint for the audio model")
    p.add_argument("-v_ckpt", "--video_checkpoint_path", type=str, help="Path to checkpoint for the video model")
    p.add_argument("-av_ckpt", "--av_checkpoint_path", type=str, help="Path to checkpoint for the audio+video model")
    p.add_argument("-m_files", "--missing_video_files",nargs='+', type=int, help="List of file numbers for incomplete video files")

    return (p.parse_args())

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
            vid_file = data_path + p_id + '_P/features/' + p_id + '_CNN_ResNet.mat'
            
            speech_file = data_path + p_id + '_P/' + p_id + '_AUDIO.wav'
            t = librosa.get_duration(path=speech_file)
            # Sampling-rate = 100
            # Preprocessing start and end times
            max_times = t*100
            df_speech = pd.read_csv(data_path + p_id + '_P/features/' + p_id + '_OpenSMILE2.3.0_egemaps.csv',sep=";")
            start_times_aud = (df_txt['Start_Time'].values*100).tolist()
            end_times_aud = (df_txt['End_Time'].values*100).tolist()

            x_aud = 1
            y_aud = len(start_times_aud)
            
            while x_aud < y_aud:
                if start_times_aud[x_aud-1]>start_times_aud[x_aud] or start_times_aud[x_aud]>max_times:
                    del start_times_aud[x_aud]
                    del end_times_aud[x_aud]
                    x_aud = x_aud-1
                    y_aud = y_aud-1
                if end_times_aud[x_aud-1]>end_times_aud[x_aud] or end_times_aud[x_aud]>max_times:
                    del start_times_aud[x_aud]
                    del end_times_aud[x_aud]
                    x_aud = x_aud-1
                    y_aud = y_aud-1
                x_aud = x_aud+1
            
            start_times_aud = [round(a) for a in start_times_aud]
            end_times_aud = [round(b) for b in end_times_aud]
            
            self.data.append([vid_file,df_txt,t,df_speech,start_times_aud,end_times_aud,float(phq_score_list[i])])
    
    # Video Preprocess
    def preprocess_vid(self,vid_file,df_txt,t):
        vid_data = sio.loadmat(vid_file)
        vid_feat = vid_data['feature']
        feat_len = len(vid_feat)
        
        # Sampling rate for video
        m_factor = feat_len/t
        
        # Preprocessing start and end times for video
        start_times_vid = (df_txt['Start_Time'].values*m_factor).tolist()
        end_times_vid = (df_txt['End_Time'].values*m_factor).tolist()

        x_vid = 1
        y_vid = len(start_times_vid)
        while x_vid < y_vid:
            if start_times_vid[x_vid-1]>start_times_vid[x_vid] or start_times_vid[x_vid]>feat_len:
                del start_times_vid[x_vid]
                del end_times_vid[x_vid]
                x_vid = x_vid-1
                y_vid = y_vid-1
            if end_times_vid[x_vid-1]>end_times_vid[x_vid] or end_times_vid[x_vid]>feat_len:
                del start_times_vid[x_vid]
                del end_times_vid[x_vid]
                x_vid = x_vid-1
                y_vid = y_vid-1
            x_vid = x_vid+1
        
        start_times_vid = [round(a) for a in start_times_vid]
        end_times_vid = [round(b) for b in end_times_vid]
        
        vid_feat_0 = torch.tensor(vid_feat[start_times_vid[0]:end_times_vid[0]]).unsqueeze(0)
        out_vid = torch.mean(vid_feat_0,dim=1)
        out_vid = out_vid.detach().cpu()

        for i in range(1,len(start_times_vid)):
            vid_feat_i = torch.tensor(vid_feat[start_times_vid[i]:end_times_vid[i]]).unsqueeze(0)
            # Mean pooling
            vid_mean_i = torch.mean(vid_feat_i,dim=1)
            vid_mean_i = vid_mean_i.detach().cpu()
            out_vid = torch.cat((out_vid,vid_mean_i),dim=0)
        l_vid = len(start_times_vid)
        # Maximum number of turns
        if l_vid > 120:
            l_vid = 120
        # Mask for attention layer
        mask_vid = torch.tensor([False]*l_vid)
        out_vid = out_vid[:l_vid]
        if l_vid < 120:
            z_vid = torch.zeros(120-l_vid,2048)
            out_vid = torch.cat((out_vid,z_vid),dim=0)
            mask_vid = torch.cat((mask_vid,torch.tensor([True]*(120-l_vid))))
        out_vid = F.normalize(out_vid, p=2, dim=1)
        return out_vid,mask_vid

    # Audio Preprocess
    def preprocess_aud(self,df_speech,start_times_aud,end_times_aud):
        df_mod = df_speech.iloc[:,2:]
        speech= df_mod.values.tolist()
        speech_0 = torch.tensor(speech[start_times_aud[0]:end_times_aud[0]]).unsqueeze(0)
        out_aud = torch.mean(speech_0, dim=1)
        out_aud = out_aud.detach().cpu()

        for i in range(1,len(start_times_aud)):
            speech_mod = torch.tensor(speech[start_times_aud[i]:end_times_aud[i]]).unsqueeze(0)
            mean_i_aud = torch.mean(speech_mod, dim=1)
            mean_i_aud = mean_i_aud.detach().cpu()
            out_aud = torch.cat((out_aud,mean_i_aud),dim=0)
        
        l_aud = len(start_times_aud)
        if l_aud > 120:
            l_aud = 120
        mask_aud = torch.tensor([False]*l_aud)
        out_aud = out_aud[:l_aud]
        if l_aud < 120:
            z_aud = torch.zeros(120-l_aud,23)
            out_aud = torch.cat((out_aud,z_aud),dim=0)
            mask_aud = torch.cat((mask_aud,torch.tensor([True]*(120-l_aud))))
        # out = F.normalize(out, p=2, dim=1)
        return out_aud,mask_aud
    def __getitem__(self,index):
        embedding_vid, mask_vid = self.preprocess_vid(self.data[index][0],self.data[index][1],self.data[index][2])
        embedding_aud, mask_aud = self.preprocess_aud(self.data[index][3],self.data[index][4],self.data[index][5])
        return [embedding_vid,mask_vid,embedding_aud,mask_aud,self.data[index][6]]
    def __len__(self):
        return len(self.data)

class lstm_regressor_vid(nn.Module):
    def __init__(self):
        super(lstm_regressor_vid, self).__init__()
        self.lstm_1 = nn.LSTM(2048,50,batch_first=True,bidirectional=True)
        self.attention1 = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.2)
        self.attention2 = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.2)
        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.2),
                                nn.Linear(12000,256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256,4))

    def forward(self,C,key_padding_mask):
        c_lstm,_ = self.lstm_1(C)
        c_att,_ = self.attention1(c_lstm,c_lstm,c_lstm,key_padding_mask=key_padding_mask)
        c_att2,_ = self.attention2(c_att,c_att,c_att,key_padding_mask=key_padding_mask)
        logits = self.mlp(c_att2)
        return logits,c_att2

class lstm_regressor_aud(nn.Module):
    def __init__(self):
        super(lstm_regressor_aud, self).__init__()
        self.lstm_1 = nn.LSTM(23,50,batch_first=True,bidirectional=True)
        self.attention1 = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.2)
        self.attention2 = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.2)
        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.2),
                                nn.Linear(12000,256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256,4))

    def forward(self,C,key_padding_mask):
        c_lstm,_ = self.lstm_1(C)
        c_att,_ = self.attention1(c_lstm,c_lstm,c_lstm,key_padding_mask=key_padding_mask)
        c_att2,_ = self.attention2(c_att,c_att,c_att,key_padding_mask=key_padding_mask)
        logits = self.mlp(c_att2)
        return logits,c_att2

class lstm_regressor(nn.Module):
    def __init__(self,vid_model,aud_model):
        super(lstm_regressor, self).__init__()
        self.vid_model = vid_model
        self.aud_model = aud_model

        self.cross_aud_vid = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.cross_vid_aud = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)

        self.self_aud_vid = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.self_vid_aud = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)

        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.8),
                                nn.Linear(24000,256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,4))

    def forward(self,C_vid,key_padding_mask_vid,C_aud,key_padding_mask_aud):

        _,c_att_vid = self.vid_model(C_vid,key_padding_mask_vid)

        _,c_att_aud = self.aud_model(C_aud,key_padding_mask_aud)

        c_aud_vid,_ = self.cross_aud_vid(c_att_aud,c_att_vid,c_att_vid,key_padding_mask=key_padding_mask_vid)
        c_vid_aud,_ = self.cross_vid_aud(c_att_vid,c_att_aud,c_att_aud,key_padding_mask=key_padding_mask_aud)

        c_att_aud_vid,_ = self.self_aud_vid(c_aud_vid,c_aud_vid,c_aud_vid,key_padding_mask=key_padding_mask_vid)
        c_att_vid_aud,_ = self.self_vid_aud(c_vid_aud,c_vid_aud,c_vid_aud,key_padding_mask=key_padding_mask_vid)

        c_comb = torch.cat((c_att_aud_vid,c_att_vid_aud),dim=2)

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
    # Put the model into the evaluation mode. The dropout layers are disabled during
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
        c_vid, mask_vid, c_aud, mask_aud, phq_scores = tuple(t.to(device) for t in batch)

        # Compute predictions
        with torch.no_grad():
            logits1 = model1.forward(c_vid,mask_vid,c_aud,mask_aud)
            logits2 = model2.forward(c_vid,mask_vid,c_aud,mask_aud)
            logits3 = model3.forward(c_vid,mask_vid,c_aud,mask_aud)
            logits4 = model4.forward(c_vid,mask_vid,c_aud,mask_aud)
            logits5 = model5.forward(c_vid,mask_vid,c_aud,mask_aud)
            logits6 = model6.forward(c_vid,mask_vid,c_aud,mask_aud)
            logits7 = model7.forward(c_vid,mask_vid,c_aud,mask_aud)
            logits8 = model8.forward(c_vid,mask_vid,c_aud,mask_aud)

        # Compute preds
        preds1 = torch.argmax(logits1, dim=1).flatten()
        preds2 = torch.argmax(logits2, dim=1).flatten()
        preds3 = torch.argmax(logits3, dim=1).flatten()
        preds4 = torch.argmax(logits4, dim=1).flatten()
        preds5 = torch.argmax(logits5, dim=1).flatten()
        preds6 = torch.argmax(logits6, dim=1).flatten()
        preds7 = torch.argmax(logits7, dim=1).flatten()
        preds8 = torch.argmax(logits8, dim=1).flatten()
        # Get total score
        preds = preds1+preds2+preds3+preds4+preds5+preds6+preds7+preds8
        preds_list.append(preds)
        labels_list.append(phq_scores)

    # Compute the CCC, RMSE and MAE
    preds_all = torch.cat(preds_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    val_loss_ccc = 1 - ccc_loss_fn(preds_all.float(), labels_all.float())
    val_loss_rmse = torch.sqrt(loss_fn_mse(preds_all.float(), labels_all.float()))
    val_loss_mae = mae_loss_fn(preds_all.float(),labels_all.float())

    return val_loss_ccc,val_loss_rmse,val_loss_mae

if __name__ == '__main__':

    args = cmdline_args()

    set_seed(args.seed)    # Set seed for reproducibility

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"# Using device: {device}")
    
    # Datasets
    data_train = dds('train',args.data_path,args.label_path,args.missing_video_files)
    data_val = dds('val',args.data_path,args.label_path,args.missing_video_files)
    data_test = dds('test',args.data_path,args.label_path,args.missing_video_files)
    
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
    v1.load_state_dict(torch.load(args.video_checkpoint_path + '-phq1-seed-' + str(args.seed) + '.pt'))
    v2.load_state_dict(torch.load(args.video_checkpoint_path + '-phq2-seed-' + str(args.seed) + '.pt'))
    v3.load_state_dict(torch.load(args.video_checkpoint_path + '-phq3-seed-' + str(args.seed) + '.pt'))
    v4.load_state_dict(torch.load(args.video_checkpoint_path + '-phq4-seed-' + str(args.seed) + '.pt'))
    v5.load_state_dict(torch.load(args.video_checkpoint_path + '-phq5-seed-' + str(args.seed) + '.pt'))
    v6.load_state_dict(torch.load(args.video_checkpoint_path + '-phq6-seed-' + str(args.seed) + '.pt'))
    v7.load_state_dict(torch.load(args.video_checkpoint_path + '-phq7-seed-' + str(args.seed) + '.pt'))
    v8.load_state_dict(torch.load(args.video_checkpoint_path + '-phq8-seed-' + str(args.seed) + '.pt'))
    
    v1.to(device)
    v2.to(device)
    v3.to(device)
    v4.to(device)
    v5.to(device)
    v6.to(device)
    v7.to(device)
    v8.to(device)

    # Define Audio Encoder for each Question
    a1 = lstm_regressor_aud()
    a2 = lstm_regressor_aud()
    a3 = lstm_regressor_aud()
    a4 = lstm_regressor_aud()
    a5 = lstm_regressor_aud()
    a6 = lstm_regressor_aud()
    a7 = lstm_regressor_aud()
    a8 = lstm_regressor_aud()
    
    # Load pretrained weights for audio encoder
    a1.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq1-seed-' + str(args.seed) + '.pt'))
    a2.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq2-seed-' + str(args.seed) + '.pt'))
    a3.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq3-seed-' + str(args.seed) + '.pt'))
    a4.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq4-seed-' + str(args.seed) + '.pt'))
    a5.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq5-seed-' + str(args.seed) + '.pt'))
    a6.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq6-seed-' + str(args.seed) + '.pt'))
    a7.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq7-seed-' + str(args.seed) + '.pt'))
    a8.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq8-seed-' + str(args.seed) + '.pt'))
    
    a1.to(device)
    a2.to(device)
    a3.to(device)
    a4.to(device)
    a5.to(device)
    a6.to(device)
    a7.to(device)
    a8.to(device)
    
    # Define A+V fusion models for each Question
    m1 = lstm_regressor(v1, a1)
    m2 = lstm_regressor(v2, a2)
    m3 = lstm_regressor(v3, a3)
    m4 = lstm_regressor(v4, a4)
    m5 = lstm_regressor(v5, a5)
    m6 = lstm_regressor(v6, a6)
    m7 = lstm_regressor(v7, a7)
    m8 = lstm_regressor(v8, a8)
    
    # Load pretrained weights for T+A fusion models
    m1.load_state_dict(torch.load(args.av_checkpoint_path + '-phq1-seed-' + str(args.seed) + '-ccc.pt'))
    m2.load_state_dict(torch.load(args.av_checkpoint_path + '-phq2-seed-' + str(args.seed) + '-ccc.pt'))
    m3.load_state_dict(torch.load(args.av_checkpoint_path + '-phq3-seed-' + str(args.seed) + '-ccc.pt'))
    m4.load_state_dict(torch.load(args.av_checkpoint_path + '-phq4-seed-' + str(args.seed) + '-ccc.pt'))
    m5.load_state_dict(torch.load(args.av_checkpoint_path + '-phq5-seed-' + str(args.seed) + '-ccc.pt'))
    m6.load_state_dict(torch.load(args.av_checkpoint_path + '-phq6-seed-' + str(args.seed) + '-ccc.pt'))
    m7.load_state_dict(torch.load(args.av_checkpoint_path + '-phq7-seed-' + str(args.seed) + '-ccc.pt'))
    m8.load_state_dict(torch.load(args.av_checkpoint_path + '-phq8-seed-' + str(args.seed) + '-ccc.pt'))

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
    loss_fn_mse = nn.MSELoss()
    ccc_loss_fn = ccc_loss()
    mae_loss_fn = nn.L1Loss()
    
    # Evaluate trained A+V model
    print(evaluate(m1,m2,m3,m4,m5,m6,m7,m8,test_dataloader))