import pandas as pd
import scipy.io as sio
import numpy as np
import random
import time
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from torcheval.metrics.functional import multiclass_f1_score

EPS = 1e-12

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--seed", type=int, help="Set a random seed")
    p.add_argument("-b", "--beta", type=float, help="The beta value used in weights for ImbOLL. Use 0.5 or 1 preferably")
    p.add_argument("-a", "--alpha", type=float, help="The alpha value used in distance for ImbOLL. Use 1,1.5 or 2 preferably")
    p.add_argument("-d_path", "--data_path", type=str, help="Path to data files (text transcripts, audio files, video features)")
    p.add_argument("-l_path", "--label_path", type=str, help="Path to labels, i.e., PHQ-8 scores")
    p.add_argument("-qno", "--question_number", type=int, help="Question number from 0 to 8")
    p.add_argument("-v_ckpt", "--video_checkpoint_path", type=str, help="Path to checkpoint for the video model")
    p.add_argument("-m_files", "--missing_video_files",nargs='+', type=int, help="List of file numbers for incomplete video files")
    p.add_argument("-train","--train_model", action='store_true',help="Wheather to train the model or not")

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
            # Files not complete
            if p_id_list[i] in missing_files_list:
                continue
            df_txt = pd.read_csv(data_path + p_id + '_P/' + p_id + '_Transcript.csv')
            vid_file = data_path + p_id + '_P/features/' + p_id + '_CNN_ResNet.mat'
            t = librosa.get_duration(path=data_path + p_id + '_P/' + p_id + '_AUDIO.wav')
            q_list = ['PHQ_8NoInterest','PHQ_8Depressed','PHQ_8Sleep','PHQ_8Tired','PHQ_8Appetite','PHQ_8Failure','PHQ_8Concentrating','PHQ_8Moving']
            score = int(df_scores[df_scores['Participant_ID']==p_id_int][q_list[q_no-1]].iloc[0])
            self.data.append([vid_file,df_txt,t,float(score)])
    
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
        out = F.normalize(out, p=2, dim=1)
        return out,mask
    def __getitem__(self,index):
        embedding, mask = self.preprocess(self.data[index][0],self.data[index][1],self.data[index][2])
        return [embedding,mask,self.data[index][3]]
    def __len__(self):
        return len(self.data)

class lstm_regressor(nn.Module):
    def __init__(self):
        super(lstm_regressor, self).__init__()
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
        return logits

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

# Get weights for ImbOLL function

def get_weights(q_no,label_path, beta):
    df_data_x = pd.read_csv(label_path + 'train_split.csv')
    p_id_list_x = df_data_x['Participant_ID'].tolist()
    df_scores = pd.read_csv(label_path + 'Detailed_PHQ8_Labels.csv')
    q_list = ['PHQ_8NoInterest','PHQ_8Depressed','PHQ_8Sleep','PHQ_8Tired','PHQ_8Appetite','PHQ_8Failure','PHQ_8Concentrating','PHQ_8Moving']
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    for i in range(len(p_id_list_x)):
        p_id_int_x = p_id_list_x[i]
        score = int(df_scores[df_scores['Participant_ID']==p_id_int_x][q_list[q_no-1]].iloc[0])
        if score ==0:
            x0 = x0+1
        elif score ==1:
            x1 = x1+1
        elif score ==2:
            x2 = x2+1
        elif score ==3:
            x3 = x3+1

    x_total = x0+x1+x2+x3
    w0 = x_total/x0
    w1 = x_total/x1
    w2 = x_total/x2
    w3 = x_total/x3
    w_total = torch.tensor([w0,w1,w2,w3])

    w = w_total**(beta)

    return w

#ImbOLL

def ImbOLL(logits,w,labels,alpha):
    num_classes = 4
    dist_matrix = [[w[0].item()*0,w[0].item()*1,w[0].item()*2,w[0].item()*3],[w[1].item()*1,w[1].item()*0,w[1].item()*1,w[1].item()*2],[w[2].item()*2,w[2].item()*1,w[2].item()*0,w[2].item()*1],[w[3].item()*3,w[3].item()*2,w[3].item()*1,w[3].item()*0]]
    probas = F.softmax(logits,dim=1)
    true_labels = [num_classes*[int(labels[k].item())] for k in range(len(labels))]
    label_ids = len(labels)*[[k for k in range(num_classes)]]
    distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
    distances_tensor = torch.tensor(distances,device=device, requires_grad=True)
    err = -torch.log(1-probas + EPS)*distances_tensor**(alpha)
    loss = torch.sum(err,axis=1).mean()
    return loss

def train(model, train_dataloader, data_train, data_val, v_ckpt_name, seed, w, alpha, val_dataloader, epochs=10, evaluation=False):

    # Start training loop
    print("Start training...\n")
    best_val_loss_ccc=-100
    best_val_loss = 10000
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
            logits = model.forward(c,mask)
            loss = ImbOLL(logits,w,phq_scores,alpha)
            batch_loss += loss.item()
            total_loss += (loss.item() * float(c.shape[0]))

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()

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
        avg_train_loss = total_loss / len(data_train)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss,val_acc,val_mi_f1,val_ma_f1,val_weight_f1,val_loss_ccc,val_rmse,val_mae = evaluate(model, data_val, val_dataloader, w, alpha)

            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                torch.save(model.state_dict(), v_ckpt_name + '-seed-' + str(seed) + '.pt')
            
            if(val_loss_ccc > best_val_loss_ccc):
                best_val_loss_ccc = val_loss_ccc
                torch.save(model.state_dict(), v_ckpt_name + '-seed-' + str(seed) + '-ccc.pt')

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, data_val, val_dataloader, w, alpha):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    total_val_loss = 0
    # Tracking variables
    val_accuracy = []
    preds_list = []
    labels_list = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        c, mask, phq_scores = tuple(t.to(device) for t in batch)

        # Compute predictions
        with torch.no_grad():
            logits = model.forward(c,mask)
        val_loss = ImbOLL(logits,w,phq_scores,alpha)
        total_val_loss += (val_loss.item()* float(c.shape[0]))
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == phq_scores).cpu().numpy().mean() * 100
        preds_list.append(preds)
        labels_list.append(phq_scores)
        val_accuracy.append(accuracy)

        preds = preds.detach().cpu()
        phq_scores = phq_scores.detach().cpu()

    # Compute the CCC, RMSE and MAE
    val_accuracy = np.mean(val_accuracy)
    preds_all = torch.cat(preds_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    print(preds_all)
    print(labels_all.long())
    
    val_mi_f1 = multiclass_f1_score(preds_all,labels_all.long(),num_classes=4)
    val_ma_f1 = multiclass_f1_score(preds_all,labels_all.long(),num_classes=4,average="macro")
    val_weight_f1 = multiclass_f1_score(preds_all,labels_all.long(),num_classes=4,average="weighted")
    val_loss_ccc = 1 - ccc_loss_fn(preds_all.float(), labels_all.float())
    val_loss_rmse = torch.sqrt(loss_fn_mse(preds_all.float(), labels_all.float()))
    val_mae = mae_loss_fn(preds_all.float(),labels_all.float())

    avg_val_loss = total_val_loss / len(data_val)

    return avg_val_loss,val_accuracy,val_mi_f1,val_ma_f1,val_weight_f1,val_loss_ccc,val_loss_rmse,val_mae

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
    data_train = dds('train',args.data_path,args.label_path,args.question_number,args.missing_video_files)
    data_val = dds('val',args.data_path,args.label_path,args.question_number,args.missing_video_files)
    
    # Define Model
    model = lstm_regressor()
    model.to(device)
    
    # Dataloaders
    train_dataloader = DataLoader(data_train,  batch_size=10,shuffle=True)
    val_dataloader = DataLoader(data_val,  batch_size=10)
    
    num_epochs = 50
    
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=5e-4,    # Default learning rate
                      eps=1e-8,    # Default epsilon value
                      weight_decay=1e-2)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * num_epochs
    
    # Specify loss functions/Metrics
    loss_fn_mse = nn.MSELoss()
    ccc_loss_fn = ccc_loss()
    mae_loss_fn = nn.L1Loss()

    w = get_weights(args.question_number,args.label_path,args.beta)
    
    if args.train_model:
        train(model, train_dataloader, data_train, data_val, args.video_checkpoint_path + '-phq' + str(args.question_number), args.seed, w, args.alpha, val_dataloader, epochs=num_epochs, evaluation=True)

    # Load trained model
    best_lstm_regressor = lstm_regressor()
    best_lstm_regressor.load_state_dict(torch.load(args.video_checkpoint_path +'-phq' + str(args.question_number) + '-seed-' + str(args.seed) + '.pt'))
    best_lstm_regressor.to(device)
    
    # Evaluate trained model
    print(evaluate(best_lstm_regressor,data_val,val_dataloader, w, args.alpha))