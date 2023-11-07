import torch

MAX_SENTENCE_LENGTh = 70
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' #('bert-base-uncased')   'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
