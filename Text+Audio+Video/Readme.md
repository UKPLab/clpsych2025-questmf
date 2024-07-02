## Text+Audio+Video files

 - TAV-Total.py: Here, M denotes the modalities used and belongs to one of (T,A,V,TA,TV,AV,TAV) depending on the folder. This file is used for training and evaluating the _Total_ framework. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - ```-t_ckpt```: Path for text model checkpoint file in _Total_ framework.
     - ```-a_ckpt```: Path for audio model checkpoint file in _Total_ framework.
     - ```-v_ckpt```: Path for video model checkpoint file in _Total_ framework.
     - ```-tav_ckpt```: Path for text+audio+video model checkpoint file in _Total_ framework.
     - ```-m_files```: Some of the data files are incomplete. This argument takes a list of such file numbers as input and ignores them.
<br>

For running this script:
```
python TAV-Total.py -d_path 'path to data' -l_path 'path to labels' -t_ckpt 'text checkpoint file path' -a_ckpt 'audio checkpoint file path' -v_ckpt 'video checkpoint file path' -tav_ckpt 'tav model checkpoint file path' -m_files xx yy zz
```
where xx, yy, zz denotes missing file numbers

 - TAV-questMF.py: Here, M denotes the modalities used and belongs to one of (T,A,V,TA,TV,AV,TAV) depending on the folder. This file is used to train the _QuestMF_ framework and evaluate it on the validation set. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - ```-t_ckpt```: Path for text model checkpoint file in _QuestMF_ framework.
     - ```-a_ckpt```: Path for audio model checkpoint file in _QuestMF_ framework.
     - ```-v_ckpt```: Path for video model checkpoint file in _QuestMF_ framework.
     - ```-tav_ckpt```: Path for text+audio+video model checkpoint file in _QuestMF_ framework.
     - ```-qno```: Since the _QuestMF_ framework trains 8 different models for each question, this argument inputs the question number for which the model will be trained.
     - ```-m_files```: Some of the data files are incomplete. This argument takes a list of such file numbers as input and ignores them.
<br>

For running this script for question number 8:
```
python TAV-questMF.py -d_path 'path to data' -l_path 'path to labels' -t_ckpt 'text checkpoint file path' -a_ckpt 'audio checkpoint file path' -v_ckpt 'video checkpoint file path' -tav_ckpt 'tav model checkpoint file path' -qno 8 -m_files xx yy zz
```
where xx, yy, zz denotes missing file numbers
 - TAV-questMF-eval.py: Here, M denotes the modalities used and belongs to one of (T,A,V,TA,TV,AV,TAV) depending on the folder. This file is used to evaluate the _QuestMF_ framework. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - ```-t_ckpt```: Path for text model checkpoint file in _QuestMF_ framework.
     - ```-a_ckpt```: Path for audio model checkpoint file in _QuestMF_ framework.
     - ```-v_ckpt```: Path for video model checkpoint file in _QuestMF_ framework.
     - ```-tav_ckpt```: Path for text+audio+video model checkpoint file in _QuestMF_ framework.
     - ```-m_files```: Some of the data files are incomplete. This argument takes a list of such file numbers as input and ignores them.
For running this script:
```
python TAV-questMF-eval.py -d_path 'path to data' -l_path 'path to labels' -t_ckpt 'text checkpoint file path' -a_ckpt 'audio checkpoint file path' -v_ckpt 'video checkpoint file path' -tav_ckpt 'tav model checkpoint file path' -m_files xx yy zz
```
where xx, yy, zz denotes missing file numbers
