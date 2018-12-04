import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #???

def importation_training_data_as_data_frame():
    '''
    Importe les données sous la forme d'un data frame pandas
    :return:
    '''
    file_path="starterkitdata"
    data_frame = pd.read_csv(file_path+"/train.txt", sep='\t', header=0)

    return data_frame


def raw_data_frame_to_corpus_and_labels(data_frame):
    partial_df=data_frame[["turn1","turn2","turn3"]]

    corpus=transform_partial_data_frame_to_corpus(partial_df)
    labels=data_frame["label"].values
    return corpus,labels



def transform_partial_data_frame_to_corpus(partial_data_frame):
    '''
    On retourne une list avec les 3 turns séparés d'un espace
    :param partial_data_frame: data frame pandas avec colonnes turn1 turn2 et turn3
    :return:
    '''

    corpus=list(( partial_data_frame.loc[:,"turn1"] + " " +partial_data_frame.loc[:,"turn2"] + " " + partial_data_frame.loc[:,"turn3"]).values)

    return corpus




def creation_corpus_training_and_labels():
    '''
    combine toutes les fonctions en une seule
    :return:
    '''
    df=importation_training_data_as_data_frame()
    return raw_data_frame_to_corpus_and_labels(df)

######################### Pas utile pour l'instant##################################
def importation_corpus_test():
    file_path = "starterkitdata"
    data_frame = pd.read_csv(file_path + "/devwithoutlabels.txt", sep='\t', header=0)

    return data_frame

def raw_data_frame_to_corpus(data_frame):

    partial_df = data_frame[["turn1", "turn2", "turn3"]]

    corpus = transform_partial_data_frame_to_corpus(partial_df)
    return corpus


def creation_corpus_test():
    df=importation_corpus_test()
    corpus=raw_data_frame_to_corpus(df)
    return corpus
