import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import HubertModel,Wav2Vec2FeatureExtractor
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
    p.add_argument("-qno", "--question_number", type=int, help="Question number from 0 to 8")
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
    def __init__(self,split,data_path,label_path,q_no,missing_files_list):
        super(dds,self).__init__()
        if split == 'train':
            df_data = pd.read_csv(label_path + 'train_split.csv')
        elif split == 'val':
            df_data = pd.read_csv(label_path + 'dev_split.csv')
        else:
            raise Exception(f"wrong split: {split}")
        self.data = []
        df_scores = pd.read_csv(label_path + 'Detailed_PHQ8_Labels.csv')
        p_id_list = df_data['Participant_ID'].tolist()
        for i in range(len(p_id_list)):
            p_id = str(p_id_list[i])
            p_id_int = p_id_list[i]
            # Files not complete for video
            if p_id_list[i] in missing_files_list:
                continue
            df_txt = pd.read_csv(data_path + p_id + '_P/' + p_id + '_Transcript.csv')
            vid_file = data_path + p_id + '_P/features/' + p_id + '_CNN_ResNet.mat'
            
            speech_file = data_path + p_id + '_P/' + p_id + '_AUDIO.wav'
            t = librosa.get_duration(path=speech_file)
            # Sampling-rate = 16000 for audio
            # Preprocessing start and end times for audio
            max_times = t*16000
            start_times_aud = (df_txt['Start_Time'].values*16000).tolist()
            end_times_aud = (df_txt['End_Time'].values*16000).tolist()

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
            q_list = ['PHQ_8NoInterest','PHQ_8Depressed','PHQ_8Sleep','PHQ_8Tired','PHQ_8Appetite','PHQ_8Failure','PHQ_8Concentrating','PHQ_8Moving']
            score = int(df_scores[df_scores['Participant_ID']==p_id_int][q_list[q_no-1]].iloc[0])
            self.data.append([vid_file,df_txt,t,speech_file,start_times_aud,end_times_aud,score])
    
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
        return out_vid,mask_vid

    # Audio Preprocess
    def preprocess_aud(self,speech_file,start_times_aud,end_times_aud):
        # Sample at original sr
        speech_org, org_sr = librosa.load(speech_file,sr=None)
        # Resample with sr = 16000
        speech = librosa.resample(speech_org, orig_sr=org_sr, target_sr=16000)
        speech_0 = speech[start_times_aud[0]:end_times_aud[0]]
        input_values = processor_aud(speech_0, sampling_rate=16000, return_tensors="pt").to(device)
        hidden_states = embedder_aud(input_values.input_values).last_hidden_state
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
                                nn.Linear(256,1))

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

def train(model, train_dataloader, av_ckpt_path, val_dataloader=None, epochs=10, evaluation=False):

    # Start training loop
    print("Start training...\n")
    best_val_loss_ccc = -100
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            c_vid, mask_vid, c_aud, mask_aud, phq_scores = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return predictions.
            preds = model.forward(c_vid,mask_vid,c_aud,mask_aud)
            # Compute loss and accumulate the loss values
            loss = loss_fn(preds.squeeze(1), phq_scores.float())
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss_ccc,_,_ = evaluate(model, val_dataloader)
            if(val_loss_ccc > best_val_loss_ccc):
                best_val_loss_ccc = val_loss_ccc
                torch.save(model.state_dict(), av_ckpt_path)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss_ccc:^10.6f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    preds_list = []
    labels_list = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        c_vid, mask_vid, c_aud, mask_aud, phq_scores = tuple(t.to(device) for t in batch)

        # Compute predictions
        with torch.no_grad():
            preds = model.forward(c_vid,mask_vid,c_aud,mask_aud)

        preds_list.append(preds.squeeze(1))
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
    processor_aud = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    embedder_aud = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    
    # Datasets
    data_train = dds('train',args.data_path,args.label_path,args.question_number,args.missing_video_files)
    data_val = dds('val',args.data_path,args.label_path,args.question_number,args.missing_video_files)
    
    # Define Video Encoder
    pretrain_vid_model = lstm_regressor_vid()
    # Load trained video model
    pretrain_vid_model.load_state_dict(torch.load(args.video_checkpoint_path + '-phq' + str(args.question_number) + '-sr-parameters.pt'))
    pretrain_vid_model.to(device)
    
    # Define Audio Encoder
    pretrain_aud_model = lstm_regressor_aud()
    # Load trained audio model
    pretrain_aud_model.load_state_dict(torch.load(args.audio_checkpoint_path + '-phq' + str(args.question_number) + '-sr16k-parameters.pt'))
    pretrain_aud_model.to(device)
    
    # Define AV Model
    model = lstm_regressor(pretrain_vid_model, pretrain_aud_model)
    model.to(device)
    
    # Dataloaders
    train_dataloader = DataLoader(data_train,  batch_size=10,shuffle=True)
    val_dataloader = DataLoader(data_val,  batch_size=10)
    
    num_epochs = 20                  
    
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=5e-4,    # Default learning rate
                      eps=1e-8,    # Default epsilon value
                      )
    
    # Total number of training steps
    total_steps = len(train_dataloader) * num_epochs
    
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)

    # Specify loss functions/Metrics
    ccc_loss_fn = ccc_loss()
    loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()
    
    # train(model, train_dataloader, args.av_checkpoint_path + '-phq' + str(args.question_number) + '-sr-parameters.pt', val_dataloader, epochs=num_epochs, evaluation=True)
    
    # Load trained AV model
    best_lstm_regressor = lstm_regressor(pretrain_vid_model, pretrain_aud_model)
    best_lstm_regressor.load_state_dict(torch.load(args.av_checkpoint_path + '-phq' + str(args.question_number) + '-sr-parameters.pt'))
    best_lstm_regressor.to(device)
    
    # Evaluate trained AV model
    print(evaluate(best_lstm_regressor,val_dataloader))