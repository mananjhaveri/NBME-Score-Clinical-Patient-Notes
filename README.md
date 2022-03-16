# NBME-Score-Clinical-Patient-Notes

### Abstract

When standardized exams are conducted, usually there is an answer key which contains a list of important points that need to be covered in the answer. The objective of our task is to use text mining methods to automate the process of detecting those important points in the answers. The manual process is tedious and exhaustive and there is scope for human bias. `Automating this process can help to generalize the scoring method so everyone is scored with the same strategy without bias with instant feedback.` Moreover, this can help students in remote areas or without access to professional teachers to get adequate feedback.

### Strategy

To combat this task and develop a proof-of-concept, we will be using [this](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/data) dataset.

**1)** **About the data**  

The text data presented here is from the USMLE Step 2 Clinical Skills examination, a medical licensure exam. This exam measures a trainee's ability to recognize pertinent clinical facts during encounters with standardized patients.

During this exam, each test taker sees a Standardized Patient, a person trained to portray a **clinical case**. After interacting with the patient, the test taker documents the relevant facts of the encounter in a **patient note**. Each patient note is scored by a trained physician who looks for the presence of certain key concepts or **features** relevant to the case as described in a rubric. The goal of this task is to develop an automated way of identifying the relevant features within each patient note, with a special focus on the patient history portions of the notes where the information from the interview with the standardized patient is documented.

**2)** **Approach**

We will be treating this task as a `multi-span question answering` method. The features/important points that need to be in the answer will be treated as the question. The patient history taken by the candidate will be the context for this task. The labels will be an annoted span from the context. We will be using GPU-powered state-of-the-art transformers fort this task.

### Research Methodology

Our primary trianing approach is based off of a paper called [`A Simple and Effective Model for Answering Multi-span Questions`](https://arxiv.org/pdf/1909.13375.pdf). It explains how to prepare the data for multi-span question answering and how to model it to use transformers.

Moreover, we experimented with a RoBERTa model pretrained for medical related text data proposed in this paper - [`Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks`](https://arxiv.org/pdf/2004.10964.pdf). Inspired by the same paper, we also experimented with pre-training  for the Masked Language Modelling task.