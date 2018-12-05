import numpy
import pickle
import pandas as pd
from sklearn.svm import  SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

from Create_features import  Objet_De_Compte

def analyse_rapide_modeles(corpus_train, y_train, bool_ajouter_autres_features):
    """
    Analyse rapidement quelques modèles avec des paramètres déterminé arbitrairement
    :param corpus_train:
    :param y_train:
    :param bool_ajouter_autres_features:
    :return:
    """

    # Objet de compte divers
    obj_features_X_wc = Objet_De_Compte("Word counts", n_gram=1, freq_min=40)
    obj_features_X_bin = Objet_De_Compte("Binary Word counts", n_gram=1, freq_min=40)
    obj_features_X_tfidf = Objet_De_Compte("TfiDf", n_gram=1, freq_min=40)

    obj_features_X_wc.fit(corpus_train)
    obj_features_X_bin.fit(corpus_train)
    obj_features_X_tfidf.fit(corpus_train)

    if bool_ajouter_autres_features:
        X_train_wc = obj_features_X_wc.transform_and_add_all_other_features(corpus_train)
        X_train_bin = obj_features_X_bin.transform_and_add_all_other_features(corpus_train)
        X_train_tfidf = obj_features_X_tfidf.transform_and_add_all_other_features(corpus_train)

    else:
        X_train_wc = pd.DataFrame(obj_features_X_wc.transform(corpus_train).toarray())
        X_train_bin = pd.DataFrame(obj_features_X_bin.transform(corpus_train).toarray())
        X_train_tfidf = pd.DataFrame(obj_features_X_tfidf.transform(corpus_train).toarray())

    # Analyse préliminaire de quelques models
    print("Modèle avec Word Count:")
    analyse_models(X_train_wc, y_train)
    print("")
    print("Modèle avec Word Count binaire:")
    analyse_models(X_train_bin, y_train)
    print("")
    print("Modèle avec TfiDf:")
    analyse_models(X_train_tfidf, y_train)


def optimisation_hyper_parametres(corpus_train, y_train, bool_ajouter_autres_features):
    """
    but: sauvegarder dict avec {nom de modèle de classification, score, dictionnaire meilleurs paramètres classification du modèle,
    nom de type d'objet de compte, dict de paramètre d'objet de compte} du modèle avec meilleur score.
    :param corpus_train:
    :param y_train:
    :param bool_ajouter_autres_features:
    :return:
    """
    ############################## Liste de paramètres à regarder ######################################################
    #List de test pour objet de compte
    list_n_gram = [1]
    list_min_freq = [2,5,10]
    list_nom_objet_compte = ["Word counts", "Binary Word counts", "TfiDf"]

    #List SVM
    list_SVM_c=[0.1, 1, 10]
    list_SVM_kernel=["rbf", "linear"]

    #List K-PPV
    list_KPPV_k=[2, 5, 8, 13, 18, 23]
    list_KPPV_weight=["uniform", "distance"]

    #List log reg
    list_LogReg_penalty=["l1", "l2"]

    #List MLP
    list_MLP_hidden_layer_shape=[(10,), (10,10), (50,50), (100,100), (200,200),
                                 (100,100,100), (100,100,100,100),(50,50,50,50,50),
                                 (10,10,10,10,10,10,10,10)]

    ####################################################################################################################

    # On veut retourner le meilleur
    list_tous_dict = []
    for n_gram in list_n_gram:
        for min_freq in list_min_freq:
            for nom_obj_compte in list_nom_objet_compte:

                #######Objet de compte######
                obj_features_X = Objet_De_Compte(nom_obj_compte, n_gram=n_gram, freq_min=min_freq)
                obj_features_X.fit(corpus_train)

                if bool_ajouter_autres_features:
                    X_train = obj_features_X.transform_and_add_all_other_features(corpus_train).values
                else:
                    X_train = obj_features_X.transform(corpus_train).toarray()


                #######SVM########
                for par_c in list_SVM_c:
                    for par_kernel in list_SVM_kernel:
                        clf = SVC(C=par_c, kernel=par_kernel)
                        score = score_simple_concours(X_train, y_train, clf)
                        dict_par={"clf name": "SVM",
                                  "score": score,
                                  "Dict param clf": {"C": par_c,
                                                     "Kernel": par_kernel},
                                  "Nom compteur": nom_obj_compte,
                                  "Dict param compteur": {"n gram":n_gram,"freq min":min_freq}
                                  }
                        list_tous_dict.append(dict_par)
                        print(dict_par)

                #######KPPV########
                for k in list_KPPV_k:
                    for weight in list_KPPV_weight:
                        clf = KNeighborsClassifier(n_neighbors=k, weights=weight)
                        score = score_simple_concours(X_train, y_train, clf)
                        dict_par = {"clf name": "KPPV",
                                    "score": score,
                                    "Dict param clf": {"k": k, "Weight": weight},
                                    "Nom compteur": nom_obj_compte,
                                    "Dict param compteur": {"n gram": n_gram, "freq min": min_freq}}
                        list_tous_dict.append(dict_par)
                        print(dict_par)

                #######Log Reg########
                for penalty in list_LogReg_penalty:
                    clf = LogisticRegression(penalty=penalty)
                    score = score_simple_concours(X_train, y_train, clf)
                    dict_par = {"clf name": "LogReg", "score": score,
                                "Dict param clf": {"penalty": penalty},
                                "Nom compteur": nom_obj_compte,
                                "Dict param compteur": {"n gram": n_gram, "freq min": min_freq}}
                    list_tous_dict.append(dict_par)
                    print(dict_par)

                #######MLP########
                for layers_shape in list_MLP_hidden_layer_shape:
                    clf = MLPClassifier(hidden_layer_sizes=layers_shape)
                    score = score_simple_concours(X_train, y_train, clf)
                    dict_par = {"clf name": "MLP",
                                "score": score,
                                "Dict param clf": {"hidden layers sizes": layers_shape},
                                "Nom compteur": nom_obj_compte,
                                "Dict param compteur": {"n gram": n_gram, "freq min": min_freq}}
                    list_tous_dict.append(dict_par)
                    print(dict_par)


    #################################Dictionnaire du modèle avec le plus haut score#####################################
    max_score = 0
    for i in range(0, len(list_tous_dict)):
        if list_tous_dict[i]["score"] > max_score:
            max_score = list_tous_dict[i]["score"]
            dict_meilleur_model = list_tous_dict[i]

    print("\nMeilleur model:", dict_meilleur_model)

    with open('Dictionnaire_parametre_meilleur_model', 'wb') as handle:
        pickle.dump(dict_meilleur_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Liste_tous_les_dictionnaires', 'wb') as handle:
        pickle.dump(list_tous_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_model_from_dict_prameters(dict):
    """
    À partir dictionnaire de modèle, on crée le modèle, on le retourne
    :param dict:
    :return: clf, modèle de type
    """
    nom_model = dict["clf name"]
    dict_param_model = dict["Dict param clf"]

    if nom_model == "SVM":
        clf = SVC(C=dict_param_model["C"], kernel=dict_param_model["Kernel"])

    elif nom_model == "KPPV":
        clf = KNeighborsClassifier(n_neighbors=dict_param_model["k"], weights=dict_param_model["Weight"])

    elif nom_model == "LogReg":
        clf = LogisticRegression(penalty=dict_param_model["penalty"])

    elif nom_model == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=dict_param_model["hidden layers sizes"])

    return clf


def analyse_models(X, y):
    """
    Calcul le score de quelques modèles rapidement
    :param X:
    :param y:
    :return:
    """
    clf = SVC(random_state=123)
    print("Score SVM:", score_simple_concours(X.values, y, clf))

    clf = KNeighborsClassifier(5)
    print("Score K-PPV:", score_simple_concours(X.values, y, clf))

    clf = LogisticRegression(random_state=123)
    print("Score Régression logistique:", score_simple_concours(X.values, y, clf))

    clf = MLPClassifier(hidden_layer_sizes=(100,100), random_state=123)
    print("Score MLP:", score_simple_concours(X.values, y, clf))


def create_X_from_dict_parameters(dict, corpus_train, corpus_test, bool_ajouter_autres_features):
    """
    Renvoie un X_train et X_test sous forme d'array selon les paramètres dans dict qui viennent créer notre objet de compte
    :param dict:
    :param corpus_train:
    :param corpus_test:
    :param bool_ajouter_autres_features:
    :return:
    """
    nom_obj_compte = dict["Nom compteur"]
    n_gram = dict["Dict param compteur"]["n gram"]
    min_freq = dict["Dict param compteur"]["freq min"]
    obj_features_X = Objet_De_Compte(nom_obj_compte, n_gram=n_gram, freq_min=min_freq)
    obj_features_X.fit(corpus_train)

    if bool_ajouter_autres_features:
        X_train = obj_features_X.transform_and_add_all_other_features(corpus_train).values
        X_test = obj_features_X.transform_and_add_all_other_features(corpus_test).values

    else:
        X_train = obj_features_X.transform(corpus_train).toarray()
        X_test = obj_features_X.transform(corpus_test).toarray()

    return X_train, X_test


########################################## Fonction de score ###########################################################
def score_simple_concours(X, y, clf):
    '''
    Calcul le score de compétition avec k fold, k=3 à partir d'un échantillon X y et un model clf
    :return: le score moyen
    '''
    kf = KFold(n_splits=3, shuffle=True, random_state=1234)
    list_score = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        score=calcul_metric_concours(y_test, y_pred)
        list_score.append(score)

    return numpy.mean(list_score)


def calcul_metric_concours(y_real, y_pred):
    """
    Calcul le score d'évaluation utilisé pour la compétition
    :param y_real:
    :param y_pred:
    :return:
    """
    import math
    def transform_y(y, i):
        if y == i:
            return 1

        else:
            return 0

    vec_transform_y = numpy.vectorize(transform_y)

    list_tp = []
    list_fp = []
    list_fn = []

    for i in range(1, 4):
        y_pred_mod = vec_transform_y(y_pred, i)
        y_real_mod = vec_transform_y(y_real, i)


        tn, fp, fn, tp = confusion_matrix(y_real_mod, y_pred_mod).ravel()
        list_tp.append(tp)
        list_fp.append(fp)
        list_fn.append(fn)

    sum_tp = numpy.sum(list_tp)
    sum_fp = numpy.sum(list_fp)
    sum_fn = numpy.sum(list_fn)
    pu = sum_tp / (sum_tp+sum_fp)
    ru = sum_tp / (sum_tp+sum_fn)

    if math.isnan(2 / (1/pu + 1/ru)):
        return 0

    else:
        return 2 / (1/pu + 1/ru)


######################################## fonction d'apparence jolies ###################################################
def plot_good_looking_confusion_matrix(y_real, y_pred):

    #Fonction trouvée sur le net
    def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        accuracy = np.trace(cm) / float(np.sum(cm))
        #misclass = 1 - accuracy
        score_competition=calcul_metric_concours(y_real,y_pred)

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; score concours={:0.4f}'.format(accuracy, score_competition))
        plt.show()

    print(classification_report(y_real,y_pred,target_names=["others", "happy", "sad", "angry"]))
    cm=confusion_matrix(y_real,y_pred)
    plot_confusion_matrix(cm,["others", "happy", "sad", "angry"],normalize=False)


if __name__ == '__main__':
    y_pred = numpy.array([0, 3, 1, 2, 2, 0])
    y_real = numpy.array([1, 3, 1, 2, 2, 0])
    print(calcul_metric_concours(y_real, y_pred))