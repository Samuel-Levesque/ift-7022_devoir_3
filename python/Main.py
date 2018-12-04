import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)
pd.set_option('display.max_colwidth', -1)

from sklearn.model_selection import train_test_split
import pickle
import time

#Fichiers maisons
from Importation_et_traitement import creation_corpus_training_and_labels, creation_corpus_test
from Analyse_corpus import analyse_corpus_labels, analyse_custom_emojis

from Pre_processing import transforme_corpus_emoji_to_characters, labels_to_y, y_to_labels
from Model_testing import analyse_rapide_modeles, optimisation_hyper_parametres, \
    create_model_from_dict_prameters, create_X_from_dict_parameters, \
    calcul_metric_concours, plot_good_looking_confusion_matrix


############################ Options de roulage de code ################################################################
bool_faire_analyse_donnees_preliminaire = True  # Environ 240 secondes
bool_faire_analyse_rapide_modeles = False  # Environ 200 secondes
bool_faire_longue_optimisation = False  # Environ 2h30
bool_faire_test_meilleur_model = False  # Environ 240 secondes
bool_faire_prediction = False  # Environ 260 secondes
bool_print_tous_models_optimisation = False  # Rapide

bool_ajouter_autres_features = False  # Augmente considérablement les délais 0.55 secondes passe à 240 secondes

if __name__ == '__main__':

    corpus,labels = creation_corpus_training_and_labels()

    if bool_faire_analyse_donnees_preliminaire:
        # Analyse caractères spéciaux
        corpus_mod = transforme_corpus_emoji_to_characters(corpus)
        analyse_custom_emojis(corpus_mod)


        analyse_corpus_labels(corpus, labels)


    #transforme les émojis en texte
    if bool_ajouter_autres_features:
        corpus = transforme_corpus_emoji_to_characters(corpus)


    # Séparation corpus d'entrainement et corpus de test
    corpus_train, corpus_test, labels_train, labels_test = train_test_split(corpus, labels, test_size=0.8, random_state=123)

    y_train = labels_to_y(labels_train)
    y_test = labels_to_y(labels_test)


    ####################################################################################################################
    ############################# Simulation d'entrainement et tests ###################################################
    ####################################################################################################################

    if bool_faire_analyse_rapide_modeles:
        analyse_rapide_modeles(corpus_train, y_train, bool_ajouter_autres_features)


    if bool_faire_longue_optimisation:
        temps_debut = time.time()
        #Liste de parmètres directement inclus dans la fonction.
        optimisation_hyper_parametres(corpus_train, y_train, bool_ajouter_autres_features)
        print("Temps de l'optimisation en secondes:", time.time() - temps_debut)


    #On va chercher les paramètres du modèle ayant le mieux performer en validation lors de l'optimisation de paramètres.
    with open('Dictionnaire_parametre_meilleur_model', 'rb') as handle:
        dictionnaire_meilleur_parametres = pickle.load(handle)
    print("\nParamètres meilleur modèle:", dictionnaire_meilleur_parametres)
    clf = create_model_from_dict_prameters(dictionnaire_meilleur_parametres)

    if bool_faire_test_meilleur_model:
        X_train,X_test = create_X_from_dict_parameters(dictionnaire_meilleur_parametres,
                                                       corpus_train,
                                                       corpus_test,
                                                       bool_ajouter_autres_features)

        #On calcul le score et confusion matrix
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        score_final = calcul_metric_concours(y_test, y_pred)
        print("Le score final du meilleur modèle est:", score_final)

        plot_good_looking_confusion_matrix(y_test, y_pred)


    if bool_faire_prediction:

        corpus_sans_labels = creation_corpus_test()
        X_train, X_test = create_X_from_dict_parameters(dictionnaire_meilleur_parametres,
                                                        corpus,
                                                        corpus_sans_labels,
                                                        bool_ajouter_autres_features)
        y_train = labels_to_y(labels)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        labels_pred = y_to_labels(y_pred)

        print(pd.DataFrame({"Corpus test":corpus_sans_labels,
                            "Label prédit":labels_pred}))


    if bool_print_tous_models_optimisation:
        with open('Liste_tous_les_dictionnaires', 'rb') as handle:
            list_dict = pickle.load(handle)

        for dict in list_dict:
            print(dict)





