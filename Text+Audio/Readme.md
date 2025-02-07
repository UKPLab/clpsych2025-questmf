## Text+Audio files

 - TA-questMF.py: This file is used to train the _QuestMF_ framework and evaluate it on the validation set. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - ```-t_ckpt```: Path for text model checkpoint file in _QuestMF_ framework.
     - ```-a_ckpt```: Path for audio model checkpoint file in _QuestMF_ framework.
     - ```-ta_ckpt```: Path for text+audio+video model checkpoint file in _QuestMF_ framework.
     - ```-qno```: Since the _QuestMF_ framework trains 8 different models for each question, this argument inputs the question number for which the model will be trained. It takes an integer value from 1 to 8.
     - ```-m_files```: Some of the data files are missing/incomplete for a certain modality. This argument takes a list of such file numbers as input and ignores them.
     - ```-train```: Whether to train the model or not. If this argument is mentioned, the model will be trained from scratch.
<br>

For running this script to train a ta model from scratch for question number 8 and evaluating on the validation set:
```
python TA-questMF.py -d_path 'path to data' -l_path 'path to labels' -t_ckpt 'text checkpoint file path' -a_ckpt 'audio checkpoint file path' -ta_ckpt 'ta model checkpoint file path' -qno 8 -train -m_files xx yy zz
```
where xx, yy, zz denote missing/incomplete file numbers.

For running this script to evaluate on the validation set for question 8 using trained tav model loaded using a checkpoint file:
```
python TA-questMF.py -d_path 'path to data' -l_path 'path to labels' -t_ckpt 'text checkpoint file path' -a_ckpt 'audio checkpoint file path' -ta_ckpt 'ta model checkpoint file path' -qno 8 -train -m_files xx yy zz
```
where xx, yy, zz denote missing/incomplete file numbers.
 - TA-questMF-eval.py:This file is used to evaluate the _QuestMF_ framework. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - ```-t_ckpt```: Path for text model checkpoint file in _QuestMF_ framework.
     - ```-a_ckpt```: Path for audio model checkpoint file in _QuestMF_ framework.
     - ```-ta_ckpt```: Path for text+audio+video model checkpoint file in _QuestMF_ framework.
     - ```-m_files```: Some of the data files are missing/incomplete for a certain modality. This argument takes a list of such file numbers as input and ignores them.
For running this script:
```
python TA-questMF-eval.py -d_path 'path to data' -l_path 'path to labels' -t_ckpt 'text checkpoint file path' -a_ckpt 'audio checkpoint file path' -ta_ckpt 'ta model checkpoint file path' -m_files xx yy zz
```
where xx, yy, zz denote missing/incomplete file numbers.
