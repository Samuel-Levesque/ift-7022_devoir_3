from Create_features import create_data_frame, add_presence_of_characters_feature, add_pourcentage_lettre_majuscule_feature, add_sentiment_features
import re
from scipy.stats import f_oneway



def analyse_corpus_labels(corpus, labels):
    '''
    Fonction principal pour analyse.
    On ajoute des colones avec les fonctions de create features, puis on regarde les occurences avec les fonctions de statistiques
    :param corpus:
    :param labels:
    :return:
    '''
    df = create_data_frame(corpus, labels)

    ################################ Présence de caractère ###################################
    print_nb_occurence_regex(df, "❤", "Ind ❤", "Total de ❤")
    print_nb_occurence_regex(df, "💔", "Ind 💔", "Total de💔")
    print_nb_occurence_regex(df, "😍", "Ind 😍", "Total de 😍")
    print_nb_occurence_regex(df, "😞", "Ind 😞", "Total de 😞")
    print_nb_occurence_regex(df, "😂", "Ind 😂", "Total de 😂")
    print_nb_occurence_regex(df, "😡", "Ind 😡", "Total de 😡")
    print_nb_occurence_regex(df, "👍", "Ind 👍", "Total de 👍")
    print_nb_occurence_regex(df, "👎", "Ind 👎", "Total de 👎")
    print_nb_occurence_regex(df, "👌", "Ind 👌", "Total de 👌")
    print_nb_occurence_regex(df, "😺", "Ind 😺", "Total de 😺")

    ####################### caracters emojis ####################

    #Smiley pos
    print_nb_occurence_regex(df, ":\)", "Ind :)", "Total de :)")
    print_nb_occurence_regex(df, ":D", "Ind :D", "Total de :D")
    print_nb_occurence_regex(df, ";\)", "Ind ;)", "Total de ;)")
    print_nb_occurence_regex(df, "=\)", "Ind =)", "Total de =)")
    print_nb_occurence_regex(df, ":-\)", "Ind :-)", "Total de :-)")
    print_nb_occurence_regex(df, ";-\)", "Ind ;-)", "Total de ;-)")
    print_nb_occurence_regex(df, ":'\)", "Ind :')", "Total de :')")
    print_nb_occurence_regex(df, ":^\)", "Ind :^)", "Total de :^)")
    print_nb_occurence_regex(df, ":]", "Ind :]", "Total de :]")

    print_nb_occurence_regex(df, ":\)|:D|;\)|=\)|:-\)|;-\)|:'\)|:^\)|:]", "Ind smiley positif", "Total de smiley positif")

    # Smiley neg
    print_nb_occurence_regex(df, ":\(", "Ind :(", "Total de :(")
    print_nb_occurence_regex(df, ";\(", "Ind ;(", "Total de ;(")
    print_nb_occurence_regex(df, ":'\(", "Ind :'(", "Total de :'(")
    print_nb_occurence_regex(df, ":/", "Ind :/", "Total de :/")
    print_nb_occurence_regex(df, ":-/", "Ind :-/", "Total de :-/")
    print_nb_occurence_regex(df, ":-\(", "Ind :-(", "Total de :-(")
    print_nb_occurence_regex(df, ":\[", "Ind :[", "Total de :[")

    print_nb_occurence_regex(df, ":\(|;\(|=\(|:-\(|:'\(|:^\(|:\[|:/|:-/", "Ind smiley negatif","Total de smiley negatif")


    #Autre smiley
    print_nb_occurence_regex(df, "-_-", "Ind -_-", "Total de -_-")
    print_nb_occurence_regex(df, "-\.-", "Ind -.-", "Total de -.-")
    print_nb_occurence_regex(df, "\^\^", "Ind ^^", "Total de ^^")
    print_nb_occurence_regex(df, "\^-\^", "Ind ^-^", "Total de ^-^")
    print_nb_occurence_regex(df, "\^\.\^", "Ind ^.^", "Total de ^.^")

    #Ponctuation diverse
    print_nb_occurence_regex(df, "\$+", "Ind serie de $", "Total de serie de $")
    print_nb_occurence_regex(df, "\.{3,}", "Ind serie de .", "Total de serie de .")
    print_nb_occurence_regex(df, "!{3,}", "Ind serie de !", "Total de serie de !")
    print_nb_occurence_regex(df, "\?{3,}", "Ind serie de ?", "Total de serie de ?")
    print_nb_occurence_regex(df, "[!\?]{3,}", "Ind serie de ! ou ?", "Total de serie de ! ou?")


    ################################### Majuscules ######################################################
    add_pourcentage_lettre_majuscule_feature(df)
    print("\nMoyenne de pourcentage de lettres en majuscule parmi toutes les lettres")
    moyenne_par_classe(df, "Pourcentage_maj")

    ################################### Sentiments ######################################################
    add_sentiment_features(df)
    print("\nMoyenne de score de sentiment")
    moyenne_par_classe(df, "Sentiment")
    print("\nMoyenne de nombre de mots positifs")
    moyenne_par_classe(df, "Nombre_positive")
    print("\nMoyenne de nombre de mots négatifs")
    moyenne_par_classe(df, "Nombre_negative")


#Fonction de statistiques
def somme_par_classe(data_frame, nom):
    '''
    À partir d'un nom de colone, calcul la somme par classe
    :param data_frame:
    :param nom:
    :return:
    '''
    print(data_frame.groupby(["Label"])[nom].sum())


def moyenne_par_classe(data_frame, nom):
    '''
    À partir d'un nom de colone, calcul la moyenne par classe
    :param data_frame:
    :param nom:
    :return:
    '''
    print(data_frame.groupby(["Label"])[nom].mean())


#Fonction de print
def print_nb_occurence_regex(df, reg_ex, nom_colone, titre_print):
    """
    Permet de print le nombre de texts qui contiennent la reg_ex, par classe
    :param df:
    :param reg_ex:
    :param nom_colone:
    :param titre_print:
    :return:
    """
    add_presence_of_characters_feature(df, reg_ex, nom_colone)
    print("\n" + titre_print)
    somme_par_classe(df, nom_colone)


############################ Autre fonction ################
def analyse_custom_emojis(corpus):
    """
    Fonction qui print tous les caractères spéciaux de longueur de 2 à 12
    :param corpus:
    :return:
    """
    myre = re.compile(r"[^\w\s\d]{2, 12}")
    list_all_match = []

    for text in corpus:
        match = myre.findall(text)
        list_all_match.extend(match)

    list_sorted = sorted(list(set(list_all_match)))

    #Print plus agréable à lire
    for i in range(0,len(list_sorted),30):
        print(list_sorted[i:i+30])

    return list_sorted





#######################


def analyse_anova_other_features(corpus, labels, list_of_custom_emojis):
    boolean_masks = {}

    df = create_data_frame(corpus, labels)

    class_list = ["happy", "angry", "sad", "others"]
    for cls in class_list:
        boolean_masks[cls] = (labels == cls)

    retained_emojis_scores = {}

    for emoji in list_of_custom_emojis:
        add_presence_of_characters_feature(df, emoji, emoji)
        retained_emojis_scores[emoji] = f_oneway(df[emoji][boolean_masks["happy"]],
                                                 df[emoji][boolean_masks["sad"]],
                                                 df[emoji][boolean_masks["angry"]],
                                                 df[emoji][boolean_masks["others"]])

    for key, value in retained_emojis_scores:
        print(key, value)

    pass



