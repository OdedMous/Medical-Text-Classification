import sys

import torch
import numpy as np

import consts
from datasets import ContrastiveDataset, SimpleDataset
from torch.utils.data import DataLoader
from transformers import AutoModel

from data import process_raw_data
from SNN_model import BERT_Arch, SiameseNeuralNetwork
from SNN_training import train_siamese_network
from projection import construct_train_matrix, extract_prototypes, project_to_dissimilarity_space
from SVM_model import ensemble_of_classifiers
from sklearn.model_selection import train_test_split

from transformers import BertTokenizerFast


def predict(projected_test, classifiers_list, categories_order):

    pred_y = []

    for classifier in classifiers_list:
         pred_y.append(classifier.predict_proba(projected_test)[:,1])    # predict_proba returns probabiltiy for class==0 and for class==1, so we take only the probabilities of class==1

    pred_y = np.vstack(pred_y) # (num_classifiers, num_samples_test)
    highest_predictions = categories_order[np.argmax(pred_y, axis=0)]
    print(pred_y)
    print(highest_predictions)

    return highest_predictions



if __name__ == '__main__':

    data_path = sys.argv[1]

    # ------------- Data --------------------------

    data, test_unseen_categories= process_raw_data(data_path)

    train_text, temp_text, train_labels, temp_labels = train_test_split(data['description'], data['labels'],
                                                                        random_state=42,
                                                                        test_size=0.3,
                                                                        stratify=data['labels'])

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=42,
                                                                    test_size=0.5,
                                                                    stratify=temp_labels)

    unseen_train_text, unseen_test_text, unseen_train_labels, unseen_test_labels = train_test_split(
        test_unseen_categories['description'], test_unseen_categories['labels'],
        random_state=42,
        test_size=0.2,
        stratify=test_unseen_categories['labels'])



    # Tokinization

    # Load the BERT tokenizer
    model_name = consts.model_name
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # tokenize and encode sequences in the sets set
    texts = [train_text, val_text, test_text, unseen_train_text, unseen_test_text]
    tokens_texts = []

    for text in texts:
        tokens_texts.append(
            tokenizer.batch_encode_plus(text.tolist(), max_length=consts.MAX_SENTENCE_LENGTh, padding='max_length',
                                        truncation=True))

    train_tokinized, val_tokinized, test_tokinized, unseen_train_tokinized, unseen_test_tokinized = tokens_texts

    def convert_to_tensors(data, labels):
        seq = torch.tensor(data['input_ids'])
        mask = torch.tensor(data['attention_mask'])
        y = torch.tensor(labels.tolist())

        return seq, mask, y


    train_seq, train_mask, train_y = convert_to_tensors(train_tokinized, train_labels)
    val_seq, val_mask, val_y = convert_to_tensors(val_tokinized, val_labels)
    test_seq, test_mask, test_y = convert_to_tensors(test_tokinized, test_labels)

    unseen_train_seq, unseen_train_mask, unseen_train_y = convert_to_tensors(unseen_train_tokinized, unseen_train_labels)
    unseen_test_seq, unseen_test_mask, unseen_test_y = convert_to_tensors(unseen_test_tokinized, unseen_test_labels)


    train_set = ContrastiveDataset(train_seq, train_mask, train_y)
    val_set = ContrastiveDataset(val_seq, val_mask, val_y)
    test_set = ContrastiveDataset(test_seq, test_mask, test_y)

    train_set_simple = SimpleDataset(train_seq, train_mask, train_y)
    test_set_simple = SimpleDataset(test_seq, test_mask, test_y)
    unseen_train_set_simple = SimpleDataset(unseen_train_seq, unseen_train_mask, unseen_train_y)
    unseen_test_set_simple = SimpleDataset(unseen_test_seq, unseen_test_mask, unseen_test_y)

    trainLoader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    valLoader = DataLoader(val_set, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    testLoader = DataLoader(test_set, batch_size=10, shuffle=False, drop_last=False, num_workers=0)

    trainLoader_simple = DataLoader(train_set_simple, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    testLoader_simple = DataLoader(test_set_simple, batch_size=64, shuffle=False, drop_last=False, num_workers=0)
    unseen_trainLoader_simple = DataLoader(unseen_train_set_simple, batch_size=64, shuffle=False, drop_last=False,
                                           num_workers=0)
    unseen_testLoader_simple = DataLoader(unseen_test_set_simple, batch_size=64, shuffle=False, drop_last=False,
                                          num_workers=0)


    # -------------- Parametrs --------------------
    model_name = consts.model_name


    # ------------- Train SNN --------------------------

    # specify GPU
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained(
        model_name)  # ('bert-base-uncased') 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False

    # pass the pre-trained BERT to our define architecture
    bert_arch = BERT_Arch(bert)

    SNN_model = SiameseNeuralNetwork(bert_arch).to(device)
    num_epochs = 30

    train_loss_history, val_loss_history, similarities_list = train_siamese_network(SNN_model,
                                                                                    dataloaders={"train": trainLoader,
                                                                                                 "val": valLoader},
                                                                                    num_epochs=num_epochs,
                                                                                    device=device)

    non_matching_similarity, matching_similarity, val_non_matching_similarity, val_matching_similarity = similarities_list


    # ----------------- Prototypes Selection ------------------------

    train_matrix = construct_train_matrix(SNN_model, trainLoader_simple)
    prototypes_list = extract_prototypes(100, trainLoader_simple, train_labels, train_matrix)

    # ---------------- Data Projection ------------------------------

    projected_train = project_to_dissimilarity_space(trainLoader_simple, SNN_model, prototypes_list)

    # ----------------- SVM Ensemble -----------------------------------

    classifiers, categories_order = ensemble_of_classifiers(projected_train, train_labels)


    # ------------------ Test: Seen categories ---------------------
    projected_test = project_to_dissimilarity_space(testLoader_simple, SNN_model, prototypes_list)
    preds = predict(projected_test, classifiers, categories_order)

    # ------------------ Test: Unseen categories ---------------------
    unseen_train_matrix = construct_train_matrix(SNN_model, unseen_trainLoader_simple)
    unseen_prototypes_list = extract_prototypes(100, unseen_trainLoader_simple, unseen_train_labels,
                                                unseen_train_matrix)
    unseen_projected_train = project_to_dissimilarity_space(unseen_trainLoader_simple, SNN_model,
                                                            unseen_prototypes_list)
    unseen_classifiers, unseen_categories_order = ensemble_of_classifiers(unseen_projected_train, unseen_train_labels)
    unseen_projected_test = project_to_dissimilarity_space(unseen_testLoader_simple, SNN_model, unseen_prototypes_list)
    unseen_preds = predict(unseen_projected_test, unseen_classifiers, unseen_categories_order)



