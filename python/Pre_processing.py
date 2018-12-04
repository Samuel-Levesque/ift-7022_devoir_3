import emoji
import numpy


#on peut utiliser cela pour transformer nos emoticon en texte
def transforme_corpus_emoji_to_characters(corpus):
    '''
    Prends un corpus et transforme les emojis en text
    :param corpus:
    :return:
    '''
    corpus_transformed = [emoji.demojize(text) for text in corpus]

    return corpus_transformed


def labels_to_y(labels):
    def fn(label):
        if label == "others":
            y = 0

        elif label == "happy":
            y = 1

        elif label == "sad":
            y = 2

        elif label == "angry":
            y = 3

        return y

    vec_fun = numpy.vectorize(fn)

    return vec_fun(labels)


def y_to_labels(y):
    def fn(y):
        if y == 0:
            label="others"

        if y == 1:
            label="happy"

        if y == 2:
            label="sad"

        if y == 3:
            label="angry"

        return label

    vec_fun = numpy.vectorize(fn)

    return vec_fun(y)
