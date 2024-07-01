import pandas as pd
import scipy.io as sio
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from transformers import HubertModel,Wav2Vec2FeatureExtractor
import librosa
import os
import argparse
from transformers import get_linear_schedule_with_warmup

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-d_path", "--data_path", type=str, help="Path to data files (text transcripts, audio files, video features)")
    p.add_argument("-l_path", "--label_path", type=str, help="Path to labels, i.e., PHQ-8 scores")
    p.add_argument("-t_ckpt", "--text_checkpoint_path", type=str, help="Path to checkpoint for the text model")
    p.add_argument("-a_ckpt", "--audio_checkpoint_path", type=str, help="Path to checkpoint for the audio model")
    p.add_argument("-v_ckpt", "--video_checkpoint_path", type=str, help="Path to checkpoint for the video model")
    p.add_argument("-tav_ckpt", "--tav_checkpoint_path", type=str, help="Path to checkpoint for the text+audio+video model")
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
            
            speech_file = data_path + p_id + '_P/' + p_id + '_AUDIO.wav'
            t = librosa.get_duration(path=speech_file)
            # Sampling-rate = 16000 for audio
            # Preprocessing start and end times for audio
            max_times = t*16000
            start_times_aud = (df_txt['Start_Time'].values*16000).tolist()
            end_times_aud = (df_txt['End_Time'].values*16000).tolist()

            vid_file = data_path + p_id + '_P/features/' + p_id + '_CNN_ResNet.mat'

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
            
            self.data.append([txt_list,speech_file,start_times_aud,end_times_aud,vid_file,df_txt,t,float(phq_score_list[i])])
    
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

    # Audio Preprocess
    def preprocess_aud(self,speech_file,start_times_aud,end_times_aud):
        # Sample at original sr
        speech_org, org_sr = librosa.load(speech_file,sr=None)
        # Resample with sr = 16000
        speech = librosa.resample(speech_org, orig_sr=org_sr, target_sr=16000)
        speech_0 = speech[start_times_aud[0]:end_times_aud[0]]
        input_values = processor_aud(speech_0, sampling_rate=16000, return_tensors="pt").to(device)
        hidden_states = embedder_aud(input_values.input_values).last_hidden_state
        # Mean pooling
        out_aud = torch.mean(hidden_states, dim=1)
        out_aud = out_aud.detach().cpu()

        for i in range(1,len(start_times_aud)):
            speech_mod = speech[start_times_aud[i]:end_times_aud[i]]
            input_values_i = processor_aud(speech_mod, sampling_rate=16000, return_tensors="pt").to(device)
            hidden_states_i = embedder_aud(input_values_i.input_values).last_hidden_state
            # Mean pooling
            mean_i_aud = torch.mean(hidden_states_i, dim=1)
            mean_i_aud = mean_i_aud.detach().cpu()
            out_aud = torch.cat((out_aud,mean_i_aud),dim=0)
        l_aud = len(start_times_aud)
        # Maximum number of turns
        if l_aud > 120:
            l_aud = 120
        # Mask for attention layer
        mask_aud = torch.tensor([False]*l_aud)
        out_aud = out_aud[:l_aud]
        if l_aud < 120:
            z_aud = torch.zeros(120-l_aud,768)
            out_aud = torch.cat((out_aud,z_aud),dim=0)
            mask_aud = torch.cat((mask_aud,torch.tensor([True]*(120-l_aud))))
        return out_aud,mask_aud

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
        # Mean pooling
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
        return out_vid,mask_vid
        
    def __getitem__(self,index):
        embedding_txt, mask_txt = self.preprocess_txt(self.data[index][0])
        embedding_aud, mask_aud = self.preprocess_aud(self.data[index][1],self.data[index][2],self.data[index][3])
        embedding_vid, mask_vid = self.preprocess_vid(self.data[index][4],self.data[index][5],self.data[index][6])
        return [embedding_txt,mask_txt,embedding_aud,mask_aud,embedding_vid,mask_vid,self.data[index][7]]
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

class lstm_regressor_aud(nn.Module):
    def __init__(self):
        super(lstm_regressor_aud, self).__init__()
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
    def __init__(self,txt_model,aud_model,vid_model):
        super(lstm_regressor, self).__init__()
        self.txt_model = txt_model
        self.aud_model = aud_model
        self.vid_model = vid_model

        self.cross_aud_txt = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.cross_txt_aud = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.cross_vid_aud = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.cross_aud_vid = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.cross_txt_vid = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)
        self.cross_vid_txt = nn.MultiheadAttention(100, 4,batch_first=True,dropout=0.8)

        self.self_aud_txt_vid = nn.MultiheadAttention(200, 4,batch_first=True,dropout=0.8)
        self.self_vid_txt_aud = nn.MultiheadAttention(200, 4,batch_first=True,dropout=0.8)
        self.self_txt_aud_vid = nn.MultiheadAttention(200, 4,batch_first=True,dropout=0.8)


        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.8),
                                nn.Linear(72000,256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,1))


        # Freeze Text Encoder
        for param_txt in self.txt_model.parameters():
            param_txt.requires_grad = False

    def forward(self,C_txt,key_padding_mask_txt,C_aud,key_padding_mask_aud,C_vid,key_padding_mask_vid):

        _,c_att_txt = self.txt_model(C_txt,key_padding_mask_txt)

        _,c_att_aud = self.aud_model(C_aud,key_padding_mask_aud)

        _,c_att_vid = self.vid_model(C_vid,key_padding_mask_vid)

        c_aud_txt,_ = self.cross_aud_txt(c_att_aud,c_att_txt,c_att_txt,key_padding_mask=key_padding_mask_txt)
        c_txt_aud,_ = self.cross_txt_aud(c_att_txt,c_att_aud,c_att_aud,key_padding_mask=key_padding_mask_aud)
        c_vid_aud,_ = self.cross_vid_aud(c_att_vid,c_att_aud,c_att_aud,key_padding_mask=key_padding_mask_aud)
        c_aud_vid,_ = self.cross_aud_vid(c_att_aud,c_att_vid,c_att_vid,key_padding_mask=key_padding_mask_vid)
        c_txt_vid,_ = self.cross_txt_vid(c_att_txt,c_att_vid,c_att_vid,key_padding_mask=key_padding_mask_vid)
        c_vid_txt,_ = self.cross_vid_txt(c_att_vid,c_att_txt,c_att_txt,key_padding_mask=key_padding_mask_txt)

        c_aud_txt_vid = torch.cat((c_aud_txt,c_aud_vid),dim=2)
        c_vid_txt_aud = torch.cat((c_vid_txt,c_vid_aud),dim=2)
        c_txt_aud_vid = torch.cat((c_txt_aud,c_txt_vid),dim=2)

        c_att_aud_txt_vid,_ = self.self_aud_txt_vid(c_aud_txt_vid,c_aud_txt_vid,c_aud_txt_vid,key_padding_mask=key_padding_mask_aud)
        c_att_vid_txt_aud,_ = self.self_vid_txt_aud(c_vid_txt_aud,c_vid_txt_aud,c_vid_txt_aud,key_padding_mask=key_padding_mask_vid)
        c_att_txt_aud_vid,_ = self.self_txt_aud_vid(c_txt_aud_vid,c_txt_aud_vid,c_txt_aud_vid,key_padding_mask=key_padding_mask_txt)
        

        c_comb = torch.cat((c_att_aud_txt_vid,c_att_vid_txt_aud,c_att_txt_aud_vid),dim=2)
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
        c_txt, mask_txt, c_aud, mask_aud, c_vid, mask_vid, phq_scores = tuple(t.to(device) for t in batch)

        # Compute predictions
        with torch.no_grad():
            preds1 = model1.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)
            preds2 = model2.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)
            preds3 = model3.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)
            preds4 = model4.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)
            preds5 = model5.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)
            preds6 = model6.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)
            preds7 = model7.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)
            preds8 = model8.forward(c_txt,mask_txt,c_aud,mask_aud,c_vid,mask_vid)

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
    tokenizer_txt = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    embedder_txt = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
    processor_aud = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    embedder_aud = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    
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
    t1.load_state_dict(torch.load(args.text_checkpoint_path + '-phq1-parameters.pt'))
    t2.load_state_dict(torch.load(args.text_checkpoint_path + '-phq2-parameters.pt'))
    t3.load_state_dict(torch.load(args.text_checkpoint_path + '-phq3-parameters.pt'))
    t4.load_state_dict(torch.load(args.text_checkpoint_path + '-phq4-parameters.pt'))
    t5.load_state_dict(torch.load(args.text_checkpoint_path + '-phq5-parameters.pt'))
    t6.load_state_dict(torch.load(args.text_checkpoint_path + '-phq6-parameters.pt'))
    t7.load_state_dict(torch.load(args.text_checkpoint_path + '-phq7-parameters.pt'))
    t8.load_state_dict(torch.load(args.text_checkpoint_path + '-phq8-parameters.pt'))

    t1.to(device)
    t2.to(device)
    t3.to(device)
    t4.to(device)
    t5.to(device)
    t6.to(device)
    t7.to(device)
    t8.to(device)

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
    a1.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq1-sr16k-parameters.pt'))
    a2.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq2-sr16k-parameters.pt'))
    a3.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq3-sr16k-parameters.pt'))
    a4.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq4-sr16k-parameters.pt'))
    a5.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq5-sr16k-parameters.pt'))
    a6.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq6-sr16k-parameters.pt'))
    a7.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq7-sr16k-parameters.pt'))
    a8.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq8-sr16k-parameters.pt'))
    
    a1.to(device)
    a2.to(device)
    a3.to(device)
    a4.to(device)
    a5.to(device)
    a6.to(device)
    a7.to(device)
    a8.to(device)

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
    v1.load_state_dict(torch.load(args.video_checkpoint_path + '-phq1-sr-parameters.pt'))
    v2.load_state_dict(torch.load(args.video_checkpoint_path + '-phq2-sr-parameters.pt'))
    v3.load_state_dict(torch.load(args.video_checkpoint_path + '-phq3-sr-parameters.pt'))
    v4.load_state_dict(torch.load(args.video_checkpoint_path + '-phq4-sr-parameters.pt'))
    v5.load_state_dict(torch.load(args.video_checkpoint_path + '-phq5-sr-parameters.pt'))
    v6.load_state_dict(torch.load(args.video_checkpoint_path + '-phq6-sr-parameters.pt'))
    v7.load_state_dict(torch.load(args.video_checkpoint_path + '-phq7-sr-parameters.pt'))
    v8.load_state_dict(torch.load(args.video_checkpoint_path + '-phq8-sr-parameters.pt'))
    
    v1.to(device)
    v2.to(device)
    v3.to(device)
    v4.to(device)
    v5.to(device)
    v6.to(device)
    v7.to(device)
    v8.to(device)
    
    # Define T+A+V fusion models for each Question
    m1 = lstm_regressor(t1,a1,v1)
    m2 = lstm_regressor(t2,a2,v2)
    m3 = lstm_regressor(t3,a3,v3)
    m4 = lstm_regressor(t4,a4,v4)
    m5 = lstm_regressor(t5,a5,v5)
    m6 = lstm_regressor(t6,a6,v6)
    m7 = lstm_regressor(t7,a7,v7)
    m8 = lstm_regressor(t8,a8,v8)
    
    # Load pretrained weights for T+A+V fusion models
    m1.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq1-sr-parameters.pt'))
    m2.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq2-sr-parameters.pt'))
    m3.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq3-sr-parameters.pt'))
    m4.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq4-sr-parameters.pt'))
    m5.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq5-sr-parameters.pt'))
    m6.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq6-sr-parameters.pt'))
    m7.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq7-sr-parameters.pt'))
    m8.load_state_dict(torch.load(args.tav_checkpoint_path + '-phq8-sr-parameters.pt'))
    
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

    # Evaluate trained T+A+V model
    print(evaluate(m1,m2,m3,m4,m5,m6,m7,m8,test_dataloader))