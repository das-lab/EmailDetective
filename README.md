# EmailDetective
EmailDetective: An Email Authorship Identification and Verification Model

## Research paper
We present the findings of this work in the following research paper:

EmailDetective: An Email Authorship Identification And Verification Model
Yong Fang, Yue Yang, Cheng Huang Published: 13 July 2020

## Introduction
Emails are often used to illegal cybercrime today, so it is important to verify the identity of the email author. This paper proposes a general model for solving the problem of anonymous email author attribution, which can be used in email authorship identification and email authorship verification. The first situation is to find the author of an anonymous email among the many suspected targets. Another situation is to verify if an email was written by the sender. This paper extracts features from the email header and email body and analyzes the writing style and other behaviors of email authors. The behaviors of email authors are extracted through a statistical algorithm from email headers. Moreover, the author’s writing style in the email body is extracted by a sequence-to-sequence bidirectional long short-term memory (BiLSTM) algorithm. This model combines multiple factors to solve the problem of anonymous email author attribution. The experiments proved that the accuracy and other indicators of proposed model are better than other methods. In email authorship verification experiment, our average accuracy, average recall and average F1-score reached 89.9%. In email authorship identification experiment, our model’s accuracy rate is 98.9% for 10 authors, 92.9% for 25 authors and 89.5% for 50 authors.

## Reference
```
@article{fang2020emaildetective,
  title={EmailDetective: An Email Authorship Identification And Verification Model},
  author={Fang, Yong and Yang, Yue and Huang, Cheng},
  journal={The Computer Journal},
  year={2020}
}
```

## model
Download the data and import it into mongodb firstly.
dataset：https://stuscueducn-my.sharepoint.com/:u:/g/personal/2018226240011_stu_scu_edu_cn/ESiIHr-yGvhMhCDyn2k0xlcBhl6dA8njkh-Rg7FOlFSEGQ?e=Bvki8l

db name： email_db

collection name：email_content_3

You can start the project through `python model_BiLstm#2020.py`, and you can check the code comments for specific details.