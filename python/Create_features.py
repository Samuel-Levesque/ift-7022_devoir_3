from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk


# Fonctions de feature principales
class Objet_De_Compte():
    def __init__(self, type_compteur, n_gram, freq_min):
        if type_compteur == "Word counts":
            self.Vectorizer = CountVectorizer(ngram_range=(n_gram,n_gram),
                                              min_df=freq_min,
                                              stop_words="english")

        if type_compteur == "Binary Word counts":
            self.Vectorizer = CountVectorizer(ngram_range=(n_gram,n_gram),
                                              min_df=freq_min,
                                              stop_words="english",
                                              binary=True)

        if type_compteur == "TfiDf":
            self.Vectorizer = TfidfVectorizer(ngram_range=(n_gram,n_gram),
                                              min_df=freq_min,
                                              stop_words="english")

    def fit(self,training_corpus):
        self.Vectorizer.fit(training_corpus)

    def transform(self,Corpus):
        X = self.Vectorizer.transform(Corpus)
        return X

    def get_feature_names(self):
        return self.Vectorizer.get_feature_names()

    def transform_and_add_all_other_features(self, corpus):
        '''
        À partir d'une matrice X, ajouter toutes les autres features qu'on juge pertinentes
        :return:
        '''
        # Compteur Principal
        array_pricinpale = self.Vectorizer.transform(corpus).toarray()
        df_compteur_principal = pd.DataFrame(array_pricinpale, columns=self.get_feature_names())

        # Autre features
        data_frame_autres_features = create_data_frame(corpus)
        add_all_other_features(data_frame_autres_features)

        # Retire le corpus du data frame avec les autres features
        data_frame_autres_features = data_frame_autres_features.drop(columns=["Corpus"])

        # Merge les deux data frame en un seul
        X = df_compteur_principal.merge(data_frame_autres_features,
                                        how="outer",
                                        left_index=True,
                                        right_index=True)

        return X


# Corpus en data_frame
def create_data_frame(corpus, labels=None):
    '''
    À partir d'une liste de text (corpus) et de labels, crée un data frame
    :param corpus:
    :param labels: optionel
    :return:
    '''
    if type(labels) == type(None):
        return pd.DataFrame({"Corpus": corpus})

    else:
        return pd.DataFrame({"Corpus": corpus, "Label": labels})


# Fonctions d'ajout de features
def add_all_other_features(data_frame):
    '''
    Ajoute tous les features qu'on veut à un data frame qui a un corpus avec les fonctions définies plus bas
    '''
    # Pas besoin d'ajouter d'emoticon car traité en texte
    add_sentiment_features(data_frame)

    add_presence_of_characters_feature(data_frame, ":\)", "Ind :)")
    add_presence_of_characters_feature(data_frame, ":D", "Ind :D")
    add_presence_of_characters_feature(data_frame, ";\)", "Ind ;)")
    add_presence_of_characters_feature(data_frame, ":-\)", "Ind :-)")
    add_presence_of_characters_feature(data_frame, ":\)|:D|;\)|=\)|:-\)|;-\)|:'\)|:^\)|:]", "Ind smiley positif")

    add_presence_of_characters_feature(data_frame, ":\(", "Ind :(")
    add_presence_of_characters_feature(data_frame, ";\(", "Ind ;(")
    add_presence_of_characters_feature(data_frame, ":'\(", "Ind :'(")
    add_presence_of_characters_feature(data_frame, ":/", "Ind :/")
    add_presence_of_characters_feature(data_frame, ":-\(", "Ind :-(")
    add_presence_of_characters_feature(data_frame, ":\(|;\(|=\(|:-\(|:'\(|:^\(|:\[|:/|:-/", "Ind smiley negatif")

    add_presence_of_characters_feature(data_frame, "!{3,}", "Ind serie de !")
    add_presence_of_characters_feature(data_frame, "\?{3,}", "Ind serie de ?")
    add_presence_of_characters_feature(data_frame, "[!\?]{3,}", "Ind serie de ! ou ?")

    add_pourcentage_lettre_majuscule_feature(data_frame)


def add_presence_of_characters_feature(data_frame,reg_expression,col_name):
    """
    À partir du dataframe, ajoute une colonne nommé col_name. La valeur est de 1 si contient l'expression reg_ex, 0 sinon
    :param data_frame:
    :param reg_expression:
    :param col_name:
    :return:
    """
    data_frame[col_name] = data_frame["Corpus"].str.contains(reg_expression, regex=True).astype(int)


def add_pourcentage_lettre_majuscule_feature(data_frame):
    '''
    Ajoute le % de lettres majuscules contenu dans un text
    '''
    serie = data_frame['Corpus'].str.count(r'[A-Z]')/data_frame['Corpus'].str.count(r'\w')
    data_frame["Pourcentage_maj"] = serie


def add_sentiment_features(data_frame):
    """
    Ajoute le score de sentiment (score pos- score neg), le nombre de mots positifs, et le nombre de mots négatifs à notre
    data frame
    """
    def get_sentiment(text):
        return swn_polarity(text)[0]
    v_sentiment = np.vectorize(get_sentiment)

    def get_positive_count(text):
        return swn_polarity(text)[1]
    v_positive_count = np.vectorize(get_positive_count)

    def get_negative_count(text):
        return swn_polarity(text)[2]
    v_negative_count = np.vectorize(get_negative_count)

    data_frame["Sentiment"] = data_frame["Corpus"].apply(v_sentiment)
    data_frame["Nombre_positive"] = data_frame["Corpus"].apply(v_positive_count)
    data_frame["Nombre_negative"] = data_frame["Corpus"].apply(v_negative_count)


#Fonctions utilitaires sentiments
def penn_to_wn(tag):
    """
    Converti PennTreebank tags en Wordnet tags
    utile dans swn_polarity()
    """
    if tag.startswith('J'):
        return wn.ADJ

    elif tag.startswith('N'):
        return wn.NOUN

    elif tag.startswith('R'):
        return wn.ADV

    elif tag.startswith('V'):
        return wn.VERB

    return None


def swn_polarity(text):
    """
    Code inspiré d'un lien internet https://nlpforhackers.io/sentiment-analysis-intro/
    Calcul le nombre de mots positifs et négatif dans un text ainsi que le score de sentiment total du text
    """
    sentiment = 0.0
    nombre_mot_negatif = 0
    nombre_mot_positif = 0

    tokens = nltk.word_tokenize(text)
    words_tags = nltk.pos_tag(tokens)
    for word, tag in words_tags:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        if swn_synset.pos_score() - swn_synset.neg_score() >0:
            nombre_mot_positif += 1

        elif swn_synset.pos_score() - swn_synset.neg_score() <0:
            nombre_mot_negatif += 1

    return sentiment,nombre_mot_positif,nombre_mot_negatif


if __name__ == '__main__':
    pass