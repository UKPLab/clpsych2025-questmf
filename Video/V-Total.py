import pandas as pd
import scipy.io as sio
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import time
import argparse
import os
from transformers import get_linear_schedule_with_warmup

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-d_path", "--data_path", type=str, help="Path to data files (text transcripts, audio files, video features)")
    p.add_argument("-l_path", "--label_path", type=str, help="Path to labels, i.e., PHQ-8 scores")
    p.add_argument("-v_ckpt", "--video_checkpoint_path", type=str, help="Path to checkpoint for the video model")
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
            # Files not complete
            if p_id_list[i] in missing_files_list:
                continue
            df_txt = pd.read_csv(data_path + p_id + '_P/' + p_id + '_Transcript.csv')
            vid_file = data_path + p_id + '_P/features/' + p_id + '_CNN_ResNet.mat'
            t = librosa.get_duration(path=data_path + p_id + '_P/' + p_id + '_AUDIO.wav')    
                
            self.data.append([vid_file,df_txt,t,float(phq_score_list[i])])
    
    def preprocess(self,vid_file,df_txt,t):
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
        mask = torch.tensor([False]*l)
        out = out[:l]
        if l < 120:
            z = torch.zeros(120-l,2048)
            out = torch.cat((out,z),dim=0)
            mask = torch.cat((mask,torch.tensor([True]*(120-l))))
        return out,mask
    def __getitem__(self,index):
        embedding, mask = self.preprocess(self.data[index][0],self.data[index][1],self.data[index][2])
        return [embedding,mask,self.data[index][3]]
    def __len__(self):
        return len(self.data)

class lstm_regressor(nn.Module):
    def __init__(self):
        super(lstm_regressor, self).__init__()
        self.lstm_1 = nn.LSTM(2048,100,batch_first=True,bidirectional=True)
        self.attention = nn.MultiheadAttention(200, 4,batch_first=True,dropout=0.5)
        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.5),
                                nn.Linear(24000,256),
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

def train(model, train_dataloader, v_ckpt_path, val_dataloader=None, epochs=10, evaluation=False):

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
            c, mask, phq_scores = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return predictions.
            preds,_ = model.forward(c,mask)
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
                torch.save(model.state_dict(), 'vid-ckpt/ResNet-mse-ccc-shuffled-sr-parameters.pt')

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
        c, mask, phq_scores = tuple(t.to(device) for t in batch)

        # Compute predictions
        with torch.no_grad():
            preds,_ = model.forward(c,mask)
        # Compute loss
        preds_list.append(preds.squeeze(1))
        labels_list.append(phq_scores)

        preds = preds.detach().cpu()
        phq_scores = phq_scores.detach().cpu()

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

    # Datasets
    data_train = dds('train',args.data_path,args.label_path,args.missing_video_files)
    data_val = dds('val',args.data_path,args.label_path,args.missing_video_files)
    data_test = dds('test',args.data_path,args.label_path,args.missing_video_files)
    
    # Define Model
    model = lstm_regressor()
    model.to(device)
    
    # Dataloaders
    train_dataloader = DataLoader(data_train,  batch_size=10, shuffle=True)
    val_dataloader = DataLoader(data_val,  batch_size=10)
    test_dataloader = DataLoader(data_test,  batch_size=10)
    
    num_epochs = 50
    
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=1e-4,    # Default learning rate
                      eps=1e-8    # Default epsilon value
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
    
    # train(model, train_dataloader, args.video_checkpoint_path+'-sr-parameters.pt', val_dataloader, epochs=num_epochs, evaluation=True)
    
    # Load trained model
    best_lstm_regressor = lstm_regressor()
    best_lstm_regressor.load_state_dict(torch.load(args.video_checkpoint_path+'-sr-parameters.pt'))
    best_lstm_regressor.to(device)
    
    # Evaluate trained model
    print(evaluate(best_lstm_regressor,test_dataloader))