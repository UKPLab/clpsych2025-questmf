## Video files

 - V-questMF.py: This file is used to train the _QuestMF_ framework and evaluate it on the validation set. It contains the following arguments:
     - ```-s```: This argument takes the seed for the experiment as input.
     - ```-a```: This argument takes the value of the $\alpha$ hyperparameter in the _ImbOLL_ function as input. Preferable values include {1,1.5,2}.
     - ```-b```: This argument takes the value of the $\beta$ hyperparameter in the _ImbOLL_ function as input. Preferable values include {0.5,1}.
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - ```-v_ckpt```: Path for video model checkpoint file in _QuestMF_ framework.
     - ```-qno```: Since the _QuestMF_ framework trains 8 different models for each question, this argument inputs the question number for which the model will be trained. It takes an integer value from 1 to 8.
     - ```-m_files```: Some of the data files are missing/incomplete for a certain modality. This argument takes a list of such file numbers as input and ignores them.
     - ```-train```: Whether to train the model or not. If this argument is mentioned, the model will be trained from scratch.
<br>

For running this script to train a tav model from scratch for question number 8 and evaluating on the validation set:
```
python V-questMF.py -s 42 -a 1 -b 0.5 -d_path 'path to data' -l_path 'path to labels' -v_ckpt 'video checkpoint file path' -qno 8 -train -m_files xx yy zz
```
where xx, yy, zz denote missing/incomplete file numbers.

For running this script to evaluate on the validation set for question 8 using trained tav model loaded using a checkpoint file:
```
python V-questMF.py -s 42 -a 1 -b 0.5 -d_path 'path to data' -l_path 'path to labels' -v_ckpt 'video checkpoint file path' -qno 8 -train -m_files xx yy zz
```
where xx, yy, zz denote missing/incomplete file numbers.
 - V-questMF-eval.py: This file is used to evaluate the _QuestMF_ framework. It contains the following arguments:
     - ```-s```: This argument takes the seed for the experiment as input.
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - ```-v_ckpt```: Path for video model checkpoint file in _QuestMF_ framework.
     - ```-m_files```: Some of the data files are missing/incomplete for a certain modality. This argument takes a list of such file numbers as input and ignores them.
For running this script:
```
python V-questMF-eval.py -s 42 -d_path 'path to data' -l_path 'path to labels' -v_ckpt 'video checkpoint file path' -m_files xx yy zz
```
where xx, yy, zz denote missing/incomplete file numbers.
