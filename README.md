# Enhancing Depression Detection via Question-wise Modality Fusion

## Abstract

Depression is a highly prevalent and disabling condition that incurs substantial personal and societal costs. Current depression diagnosis involves determining the depression severity of a person through self-reported questionnaires or interviews conducted by clinicians. This often leads to delayed treatment and involves substantial human resources. Thus, several works have tried to automate the process using multimodal data. However, they overlooked the dynamic contribution of each modality for each question in the questionnaire, thus leading to sub-optimal fusion. In this work, we propose a novel Question-wise Modality Fusion (_QuestMF_) framework to tackle this issue. Our framework outperforms current state-of-the-art models on the E-DAIC dataset. In addition, we provide an analysis to understand each modality's contribution to each question's scoring. Our framework also offers symptom-specific insights, leading to greater interpretability and specificity in depression detection, which can facilitate more personalised interventions.

## Dataset

For this work, we use the E-DAIC dataset from the AVEC-2019 challenge. The dataset can be found here: https://dcapswoz.ict.usc.edu/

## Summary

In our work, we try to predict the PHQ-8 scores from recorded interviews in the E-DAIC dataset. A PHQ-8 questionnaire contains 8 questions about depression symptoms, each scored from 0-3 depending on how frequently a person encounters them. This gives a total score in the range of 0-24. Here we experiment with two different frameworks:

- _QuestMF_: Here, we train separate models to predict the score for each question in the questionnaire using session inputs. We sum the scores from each question to get the total questionnaire score. This framework includes 8 single modality encoders for each modality and 8 fused models corresponding to the 8 questions in a PHQ-8 questionnaire.
- _Total_: Here, we train the models to predict the total questionnaire score from the session inputs. This consists of a single modality encoder for each modality and a single fused model.

## Contact
For any questions contact: [Aishik Mandal](mailto:aishik.mandal@tu-darmstadt.de) <br>
[UKP Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) | [TU Darmstadt](https://www.tu-darmstadt.de/) 

## Creating the environment

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Code Structure

The code base contains 7 folders, each containing files for a single combination of modalities. These are elaborated as follows:

- Text: This folder contains codes using only text modality, i.e., they use textual transcript data.
- Audio: This folder contains codes using only audio modality, i.e., they use data from recorded audio files.
- Video: This folder contains codes using only video modality, i.e., they use ResNet features from the recorded videos.
- Text+Audio: This folder contains codes using both text and audio data.
- Text+Video: This folder contains codes using both text and video data.
- Audio+Video: This folder contains codes using both audio and video data
- Text+Audio+Video: This folder contains codes using all the available modalities data, i.e., text, audio and video data.

Each of these folders contains three files:

 - M-Total.py: Here, M denotes the modalities used and belongs to one of (T,A,V,TA,TV,AV,TAV) depending on the folder. This file is used for training and evaluating the _Total_ framework. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - Checkpoint files: These arguments take the checkpoints to save and load the trained models. This argument differs depending on the combination of modalities used and is further explained in the respective folders.
     - ```-m_files```: Some of the data files are missing/incomplete for a certain modality. This argument takes a list of such file numbers as input and ignores them.
     - ```-train```: Whether to train the model or not. If this argument is mentioned, the model will be trained from scratch.
 - M-questMF.py: Here, M denotes the modalities used and belongs to one of (T,A,V,TA,TV,AV,TAV) depending on the folder. This file is used to train the _QuestMF_ framework. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - Checkpoint files: These arguments take the checkpoints to save and load the trained models. This argument differs depending on the combination of modalities used and is further explained in the respective folders.
     - ```-qno```: Since the _QuestMF_ framework trains 8 different models for each question, this argument inputs the question number for which the model will be trained. It takes an integer value from 0 to 8.
     - ```-m_files```: Some of the data files are missing/incomplete for a certain modality. This argument takes a list of such file numbers as input and ignores them.
     - ```-train```: Whether to train the model or not. If this argument is mentioned, the model will be trained from scratch.
 - M-questMF-eval.py: Here, M denotes the modalities used and belongs to one of (T,A,V,TA,TV,AV,TAV) depending on the folder. This file is used to evaluate the _QuestMF_ framework. It contains the following arguments:
     - ```-d_path```: This argument takes the data path as input. The data path contains the text transcripts files, audio files and video features files.
     - ```-l_path```: This argument takes the label path as input. The label path contains the PHQ-8 scores for the test, validation and test splits. It also contains fine-grained question-wise scores for train and validation splits.
     - Checkpoint files: These arguments take the checkpoints to save and load the trained models. This argument differs depending on the combination of modalities used and is further explained in the respective folders. The checkpoint path given here should be the same as the checkpoint path given in M-questMF.py.
     - ```-m_files```: Some of the data files are missing/incomplete for a certain modality. This argument takes a list of such file numbers as input and ignores them.
<br>

**Further details on running the scripts are provided in each folder**

## Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
