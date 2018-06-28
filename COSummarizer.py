#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
    Developed by: Otávio Augusto Ferreira Sousa
    Last modified: June 27, 2018

    This file contains the implementation of a Contrastive Opinion Summarizer based on the work of Kim & Zhai(2009).

    In order to execute this program, you should define some parameters. If you don't, default values will be used.
    All these parameters are predefined in a dictionary which is identified as 'PARAMETERS' and declared in the first section of this code.

    There are two ways to set these parameters.
    You could pass them as command line arguments or simply modify the dictionary called 'PARAMETERS' in this code.
    If you'd rather use command line arguments, here is an example on how you should do it:

    $ python COSummarizer.py --dataset=sample_datasets/english/MicroMP3_design.txt --language=english --lambda=0.5 --method=CF --centroids_as_summary=False --use_hungarian_method=True --allow_repetition=False

    Please see details about the parameters below.

    DATASET  -> path and filename of the data set;
                each line of the data set must follow this format:
                [Polarity] Id - Sentence
                - Polarity: 0 if the sentence is negative, 1 if it is positive.
                - Id: id of the sentence. It goes from 1 to N for positive sentences and from 1 to M for negative ones.
                - Sentence: the opinionated sentence itself.
                If you want a few examples, please see the files in the folder 'sample_datasets/'.

    LANGUAGE -> it defines which language should be considered during program execution;
                'portuguese' and 'english' are accepted.

    LAMBDA   -> a float value in range [0, 1];
                the bigger it is, the more importance you give to the representativeness of the summary;
                the smaller, the more importance you give to the contrastiveness of the summary.

    METHOD   -> which method should be executed;
                'RF' (Representativeness-First) and 'CF' (Contrastiveness-First) are accepted.

                The method 'RF' allows some options:
                CENTROIDS_AS_SUMMARY: [boolean] represents whether to use the centroids of the aligned clusters to compose the summary or to try to find better sentences than the centroids pairs.
                USE_HUNGARIAN_METHOD: [boolean] represents whether to use the hungarian method or brute force approach to find the best alignment.

                The method 'CF' has the option:
                ALLOW_REPETITION: [boolean] represents whether it is allowed to repeat a sentence in the summary or not.

    Reference to the paper Kim & Zhai (2009):
    Hyun Duk Kim and ChengXiang Zhai, Generating Comparative Summaries of Contradictory Opinions in Text, CIKM 2009

    Reference to POS Tagger in Portuguese language:
    Fonseca, E. R. and Rosa, J.L.G. Mac-Morpho Revisited: Towards Robust Part-of-Speech Tagging. Proceedings of the 9th Brazilian Symposium in Information and Human Language Technology, 2013. p. 98-107
"""


# <------------------------>
# <-----> PARAMETERS <----->
# <------------------------>


PARAMETERS = dict()
PARAMETERS['DATASET'] = 'sample_datasets/portuguese/Smartphone_Samsung_Galaxy_S7_SM-G930_32GB_bateria.txt'
PARAMETERS['LANGUAGE'] = 'portuguese'
PARAMETERS['LAMBDA'] = '0.5'
PARAMETERS['METHOD'] = 'RF'
PARAMETERS['CENTROIDS_AS_SUMMARY'] = 'False'
PARAMETERS['USE_HUNGARIAN_METHOD'] = 'True'
PARAMETERS['ALLOW_REPETITION'] = 'False'


# <---------------------------->
# <-----> Common Modules <----->
# <---------------------------->


# basic functions and constants were used
import sys
import getopt
import math
from itertools import permutations
from operator import itemgetter

# used in data preprocessing
import unicodedata
import string
import re

# <------------------------------------------------->
# <-----> Natural Language Processing Modules <----->
# <------------------------------------------------->


# used for: tagger for English, tokenizer, stopwords lists, stemmer
import nltk
from nltk.stem import RSLPStemmer, PorterStemmer

# used for: tagger for Portuguese
import nlpnet

# <------------------------------->
# <-----> Auxiliary Modules <----->
# <------------------------------->


# clustering algorithm used in the implementation of RF method
from sklearn.cluster import AgglomerativeClustering

# used to obtain the best alignment in polynomial time (Hungarian method)
from scipy.optimize import linear_sum_assignment

# <----------------------->
# <-----> Variables <----->
# <----------------------->


# NLP variables
stemmer = dict(portuguese=RSLPStemmer(), english=PorterStemmer())
nlpnet.set_data_dir("nlpnet_data/")
nlpnet_POSTagger = nlpnet.POSTagger()

# lists of negation words
negation_words = {
    'portuguese': ["jamais", "nada", "nem", "nenhum", "ninguém", "nunca", "não", "tampouco"],
    'english': ["never", "neither", "nobody", "no", "none", "nor", "nothing", "nowhere", "not", 'n\'t']
}

# represents the summary
contrastive_pairs = [(0, 0), (1, 1)]

# structure that stores the raw and processed sentences for both polarities positive and negative
opinion_sets = {
    'raw': {'+': [], '-': []},
    'processed': {'+': [], '-': []}
}

# in order to gain efficiency timewise, if a ϕ or ψ is calculated, it's stored
# so it wouldn't be necessary to calculate it again
phi_table = {'+': {}, '-': {}}
psi_table = dict()
omega_table = dict()


# <--------------------------------->
# <-----> Auxiliary Functions <----->
# <--------------------------------->


def simplify_characters(my_string):
    """
    This function normalize the characters, transform characters into lowercase and remove both punctuation and special characters from a string.
    :param my_string: [str]
    :return: [str] returns a simplified string with only numbers, letters and spaces.
    """

    # normalize and transform characters into lowercase
    normalized_string = unicodedata.normalize('NFKD', my_string.casefold())

    # simplify characters in a way that ç becomes c, á becomes a, etc.
    # normalized_string = u"".join([c for c in normalized_string if not unicodedata.combining(c)])

    # remove punctuation and special characters
    return re.sub('[' + string.punctuation + ']', '', normalized_string)


def show_summary(cell_length=30):
    """
    Function that exhibits the summary in form of a table with two columns: positive sentences and negative sentences.
    :param cell_length: [int] indicates how many characters will be in each cell at most.
    :return: none.
    """

    column_headers = {
        'portuguese': ['Positiva', 'Negativa'],
        'english': ['Positive', 'Negative']
    }

    header_line = "+" + ("-" * cell_length) + "+" + ("-" * cell_length) + "+"

    header = ""
    header += header_line
    header += '\n'
    header += "|" + (' ' * ((cell_length - len(column_headers[PARAMETERS['LANGUAGE']][0])) // 2)) + \
              column_headers[PARAMETERS['LANGUAGE']][0] + (
                      ' ' * int(math.ceil(((cell_length - len(column_headers[PARAMETERS['LANGUAGE']][0])) / 2))))
    header += "|" + (' ' * ((cell_length - len(column_headers[PARAMETERS['LANGUAGE']][1])) // 2)) + \
              column_headers[PARAMETERS['LANGUAGE']][1] + (
                      ' ' * int(math.ceil(((cell_length - len(column_headers[PARAMETERS['LANGUAGE']][1])) / 2)))) + "|"
    header += '\n'
    header += header_line

    print(header)

    for (pos, neg) in contrastive_pairs:
        pos = opinion_sets['raw']['+'][pos]
        neg = opinion_sets['raw']['-'][neg]
        pos_idx = 0
        neg_idx = 0

        while pos_idx < len(pos) or neg_idx < len(neg):
            print("|", end="")

            coln = 0
            while pos_idx < len(pos) and coln < cell_length:
                print(pos[pos_idx], end="")
                pos_idx += 1
                coln += 1

            while coln < cell_length:
                print(" ", end="")
                coln += 1

            print("|", end="")

            coln = 0
            while neg_idx < len(neg) and coln < cell_length:
                print(neg[neg_idx], end="")
                neg_idx += 1
                coln += 1

            while coln < cell_length:
                print(" ", end="")
                coln += 1

            print("|")

        print(header_line)
    print("\n")


# <--------------------------->
# <-----> COS Functions <----->
# <--------------------------->


def omega(u, v):
    """
    Term Similarity Function (ω) of two words.
    :param u: [str] word to be compared.
    :param v: [str] word to be compared.
    :return: [float] value in range [0, 1].
    """

    if u not in omega_table.keys():
        omega_table[u] = stemmer[PARAMETERS['LANGUAGE']].stem(u)

    if v not in omega_table.keys():
        omega_table[v] = stemmer[PARAMETERS['LANGUAGE']].stem(v)

    # using Word Overlap
    if omega_table[u] == omega_table[v]:
        return 1.0
    else:
        return 0.0


def phi(sentence1, sentence2, polarity):
    """
    Given two opinionated sentences with the same polarity, the content similarity function φ(s1,s2) ∈ [0,1] measures the overall content similarity of s1 and s2.
    :param sentence1: [int] index of the sentence s1 to be compared.
    :param sentence2: [int] index of the sentence s2 to be compared.
    :param polarity: [str] '+' or '-', i.e., whether the sentences belong to the positive or negative opinion set.
    :return: [float] a value in range [0, 1].
    """

    sentence1, sentence2 = min(sentence1, sentence2), max(sentence1, sentence2)
    idx = ','.join([str(sentence1), str(sentence2)])

    # if ϕ(sentence1, sentence2) isn't calculated yet
    if idx not in phi_table[polarity].keys():

        # obtain the sentences themselves
        sentence1 = opinion_sets['processed'][polarity][sentence1]
        sentence2 = opinion_sets['processed'][polarity][sentence2]

        # implementing the formula in the paper
        sum_sentence1 = 0.0
        for word1 in sentence1:
            max_omega = 0.0
            for word2 in sentence2:
                max_omega = max(max_omega, omega(word1, word2))
            sum_sentence1 += max_omega

        sum_sentence2 = 0.0
        for word2 in sentence2:
            max_omega = 0.0
            for word1 in sentence1:
                max_omega = max(max_omega, omega(word1, word2))
            sum_sentence2 += max_omega

        phi_table[polarity][idx] = float(sum_sentence1 + sum_sentence2) / (len(sentence1) + len(sentence2))

    return phi_table[polarity][idx]


def psi(sentence1, sentence2):
    """
    Given two opinionated sentences s1 and s2 with opposite polarities, the contrastive similarity function ψ(s1,s2) ∈ [0,1] measures the similarity of s1 and s2 excluding their difference in sentiment.
    :param sentence1: [int] index of the positive sentence s1 to be compared.
    :param sentence2: [int] index of the negative sentence s2 to be compared.
    :return: [float] a value in range [0, 1].
    """

    idx = ','.join([str(sentence1), str(sentence2)])

    # if ψ(sentence1, sentence2) isn't calculated yet
    if idx not in psi_table.keys():

        # obtain the sentence itself
        sentence1 = opinion_sets['processed']['+'][sentence1]
        sentence2 = opinion_sets['processed']['-'][sentence2]

        # removing negation words from both sentences
        sentence1 = [word for word in sentence1 if (word not in negation_words[PARAMETERS['LANGUAGE']])]
        sentence2 = [word for word in sentence2 if (word not in negation_words[PARAMETERS['LANGUAGE']])]

        # removing adjectives from both sentences
        if PARAMETERS['LANGUAGE'] == 'english':
            sentence1 = [token for (token, tag) in nltk.pos_tag(sentence1, tagset='universal') if (tag != 'ADJ')]
            sentence2 = [token for (token, tag) in nltk.pos_tag(sentence2, tagset='universal') if (tag != 'ADJ')]
        elif PARAMETERS['LANGUAGE'] == 'portuguese':
            sentence1 = [token for (token, tag) in nlpnet_POSTagger.tag_tokens(sentence1, return_tokens=True) if
                         (tag != 'ADJ')]
            sentence2 = [token for (token, tag) in nlpnet_POSTagger.tag_tokens(sentence2, return_tokens=True) if
                         (tag != 'ADJ')]

        # implementing the formula in the paper
        sum_sentence1 = 0.0
        for word1 in sentence1:
            max_omega = 0.0
            for word2 in sentence2:
                max_omega = max(max_omega, omega(word1, word2))
            sum_sentence1 += max_omega

        sum_sentence2 = 0.0
        for word2 in sentence2:
            max_omega = 0.0
            for word1 in sentence1:
                max_omega = max(max_omega, omega(word1, word2))
            sum_sentence2 += max_omega

        if len(sentence1) + len(sentence2) == 0:
            psi_table[idx] = 0.0
        else:
            psi_table[idx] = float(sum_sentence1 + sum_sentence2) / (len(sentence1) + len(sentence2))

    return psi_table[idx]


def representativeness_first():
    """
    Representativeness-first approximation to solve the Contrastive Opinion Summarization problem (COS).
    """

    # amount of sentence pairs we want in the summary
    k = 1 + int(math.floor(math.log2(len(opinion_sets['processed']['+']) + len(opinion_sets['processed']['-']))))

    # calculating the distance matrices; soon it will be used for agglomerative clustering
    # basically, they are like this: distance_matrix[i, j] = 1.0 - phi(i, j)
    distance_matrix_x = [[(1.0 - phi(i, j, '+')) for j in range(len(opinion_sets['processed']['+']))] for i in range(len(opinion_sets['processed']['+']))]
    distance_matrix_y = [[(1.0 - phi(i, j, '-')) for j in range(len(opinion_sets['processed']['-']))] for i in range(len(opinion_sets['processed']['-']))]

    # each one below is gonna be a list of clusters (list of lists)
    clusters_x = [list() for i in range(k)]
    clusters_y = [list() for i in range(k)]

    # defining the clustering model that is going to be used
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=k, linkage='average')

    # obtaining the clusters for X and Y
    model_labels_x = model.fit_predict(distance_matrix_x)
    model_labels_y = model.fit_predict(distance_matrix_y)

    for i in range(len(opinion_sets['processed']['+'])):
        clusters_x[model_labels_x[i]].append(i)
    for j in range(len(opinion_sets['processed']['-'])):
        clusters_y[model_labels_y[j]].append(j)

    # structures to store the centroids for each cluster
    centroids_x = [-1 for i in range(k)]
    centroids_y = [-1 for i in range(k)]

    # finding the centroids for clusters_x
    for cluster_id, cluster in enumerate(clusters_x):
        best_value = -math.inf
        best_sentence = -1

        for sentence1 in cluster:

            value = 0.0
            for sentence2 in cluster:
                value += phi(sentence1, sentence2, '+')

            if value > best_value:
                best_value = value
                best_sentence = sentence1

        centroids_x[cluster_id] = best_sentence

    # finding the centroids for clusters_y
    for cluster_id, cluster in enumerate(clusters_y):
        best_value = -math.inf
        best_sentence = -1

        for sentence1 in cluster:

            value = 0.0
            for sentence2 in cluster:
                value += phi(sentence1, sentence2, '-')

            if value > best_value:
                best_value = value
                best_sentence = sentence1

        centroids_y[cluster_id] = best_sentence

    # finding the best alignment
    best_alignment = -1

    if PARAMETERS['USE_HUNGARIAN_METHOD']:
        # using the hungarian method in order to find the best alignment for the clusters
        row_ind, col_ind = linear_sum_assignment(
            [[-1.0 * psi(centroids_x[U], centroids_y[V]) for V in range(k)] for U in range(k)])
        best_alignment = list(zip(row_ind, col_ind))
    else:
        # using brute force in order to find the best alignment for the clusters
        best_value = -math.inf
        for permutation in permutations(range(k)):

            # turning alignment into a list of pairs like: [(0, 1), (1, 2), (2, 0)]
            alignment = list(zip(range(k), list(permutation)))

            value = 0.0
            for U, V in alignment:
                value += psi(centroids_x[U], centroids_y[V])

            if value > best_value:
                best_value = value
                best_alignment = alignment

    if PARAMETERS['CENTROIDS_AS_SUMMARY']:
        # using the centroids pairs to compose the summary
        for U, V in best_alignment:
            contrastive_pairs.append((centroids_x[U], centroids_y[V]))
    else:
        # choosing better sentence pairs than centroid pairs to compose the summary
        for U, V in best_alignment:

            list_pairs = list()
            for ui in clusters_x[U]:
                for vi in clusters_y[V]:
                    list_pairs.append((ui, vi, psi(ui, vi)))

            # sorting all the pairs in each cluster pair in descending order of ψ()
            list_pairs.sort(key=itemgetter(2), reverse=True)

            # obtaining the sentence pair that gives the highest gi value
            best_value = -math.inf
            best_pair = (-1, -1)
            for ui, vi, psi_value in list_pairs:

                sum_u = 0.0
                for x in clusters_x[U]:
                    sum_u += phi(x, ui, '+')
                sum_u /= len(clusters_x[U])

                sum_v = 0.0
                for y in clusters_y[V]:
                    sum_v += phi(y, vi, '-')
                sum_v /= len(clusters_y[V])

                value = PARAMETERS['LAMBDA'] * (sum_u + sum_v) + ((1.0 - PARAMETERS['LAMBDA']) / k) * psi_value

                if value > best_value:
                    best_value = value
                    best_pair = (ui, vi)

                # break loop when reach the centroid pair
                if (ui, vi) == (centroids_x[U], centroids_y[V]):
                    break

            # adding the sentence pair with highest gi to the summary
            contrastive_pairs.append(best_pair)


def contrastiveness_first():
    """
    Contrastiveness-first approximation to solve the Contrastive Opinion Summarization problem (COS).
    """

    # amount of sentence pairs we want in the summary
    k = 1 + int(math.floor(math.log2(len(opinion_sets['processed']['+']) + len(opinion_sets['processed']['-']))))

    # it will be used to mark the pairs that are in the summary already
    pair_already_chosen = [[False for sY in range(len(opinion_sets['processed']['-']))] for sX in range(len(opinion_sets['processed']['+']))]  # |X| x |Y|
    u_chosen, v_chosen = [], []

    if not PARAMETERS['ALLOW_REPETITION']:
        u_chosen = [False for sX in range(len(opinion_sets['processed']['+']))]
        v_chosen = [False for sY in range(len(opinion_sets['processed']['-']))]

    # structures used to find the sets X_ui and Y_vi defined in the paper
    max_similarity_x = [0.0 for sX in opinion_sets['processed']['+']]
    max_similarity_y = [0.0 for sY in opinion_sets['processed']['-']]

    # finding the pair with the highest value of ψ
    best_pair = (0, 0)
    for i in range(len(opinion_sets['processed']['+'])):
        for j in range(len(opinion_sets['processed']['-'])):
            if psi(i, j) > psi(best_pair[0], best_pair[1]):
                best_pair = (i, j)

    # initially, add the pair with the highest value of ψ to summary
    contrastive_pairs.append(best_pair)

    # mark this pair as chosen already, so it will be ignored later
    pair_already_chosen[best_pair[0]][best_pair[1]] = True

    if not PARAMETERS['ALLOW_REPETITION']:
        u_chosen[best_pair[0]] = True
        v_chosen[best_pair[1]] = True

    # computing the best φ values for each sentence, initially
    for i in range(len(opinion_sets['processed']['+'])):
        max_similarity_x[i] = max(max_similarity_x[i], phi(best_pair[0], i, '+'))
    for j in range(len(opinion_sets['processed']['-'])):
        max_similarity_y[j] = max(max_similarity_y[j], phi(best_pair[1], j, '-'))

    # after choosing the ﬁrst pair, we will iteratively choose a pair to maximize the "gain function"
    for k_i in range(1, k):
        best_value = -math.inf
        best_pair = (-1, -1)

        # testing the pair (i, j)
        for i in range(len(opinion_sets['processed']['+'])):
            for j in range(len(opinion_sets['processed']['-'])):
                if pair_already_chosen[i][j]:
                    continue
                if not PARAMETERS['ALLOW_REPETITION']:
                    if u_chosen[i] or v_chosen[j]:
                        continue

                # compute the gain function for this pair (i, j)
                sum_in_x, sum_in_y = 0.0, 0.0

                # These two following loops are computing the sums in gain function, considering the sets X_ui and Y_vi
                for m in range(len(opinion_sets['processed']['+'])):
                    if phi(m, i, '+') > max_similarity_x[m]:
                        sum_in_x += phi(m, i, '+')
                sum_in_x /= len(opinion_sets['processed']['+'])

                for m in range(len(opinion_sets['processed']['-'])):
                    if phi(m, j, '-') > max_similarity_y[m]:
                        sum_in_y += phi(m, j, '-')
                sum_in_y /= len(opinion_sets['processed']['-'])

                value = PARAMETERS['LAMBDA'] * (sum_in_x + sum_in_y) + ((1.0 - PARAMETERS['LAMBDA']) / k) * psi(i, j)

                if value > best_value:  # the pair that maximizes the gain function will be added to the summary
                    best_value = value
                    best_pair = (i, j)

        # add the pair with the highest gain function to summary
        contrastive_pairs.append(best_pair)

        # mark this pair as chosen already, so it will be ignored in the next iterations
        pair_already_chosen[best_pair[0]][best_pair[1]] = True

        if not PARAMETERS['ALLOW_REPETITION']:
            u_chosen[best_pair[0]] = True
            v_chosen[best_pair[1]] = True

        # remember the best φ values for each of all sentences given by the i−1 already chosen sentence pairs at each step
        for i in range(len(opinion_sets['processed']['+'])):
            max_similarity_x[i] = max(max_similarity_x[i], phi(best_pair[0], i, '+'))
        for j in range(len(opinion_sets['processed']['-'])):
            max_similarity_y[j] = max(max_similarity_y[j], phi(best_pair[1], j, '-'))


def main(argv):
    global PARAMETERS
    global contrastive_pairs, opinion_sets, phi_table, psi_table, omega_table

    # define the parameters according to what was given in the command line arguments;
    # if there are none, the program will consider the predefined parameters in the beginning of this code.
    try:
        opts, args = getopt.getopt(argv, "", ["dataset=", "language=", "lambda=", "method=", "centroids_as_summary=",
                                              "use_hungarian_method=", "allow_repetition="])
        for opt, arg in opts:
            PARAMETERS[opt[2:].upper()] = arg
        PARAMETERS['LAMBDA'] = float(PARAMETERS['LAMBDA'])
        PARAMETERS['CENTROIDS_AS_SUMMARY'] = (PARAMETERS['CENTROIDS_AS_SUMMARY'] == 'True')
        PARAMETERS['USE_HUNGARIAN_METHOD'] = (PARAMETERS['USE_HUNGARIAN_METHOD'] == 'True')
        PARAMETERS['ALLOW_REPETITION'] = (PARAMETERS['ALLOW_REPETITION'] == 'True')
    except (getopt.GetoptError, ValueError):
        print('Error: please see the documentation in order to provide parameters correctly.')
        sys.exit(2)

    # resetting structures
    contrastive_pairs.clear()
    opinion_sets['processed']['+'].clear()
    opinion_sets['processed']['-'].clear()
    opinion_sets['raw']['+'].clear()
    opinion_sets['raw']['-'].clear()
    phi_table = {'+': dict(), '-': dict()}
    psi_table = dict()
    omega_table = dict()

    # <--------------------------------------->
    # <--------> loading the dataset <-------->
    # <--------------------------------------->

    with open(PARAMETERS['DATASET'], encoding='utf-8') as fp_dataset:
        for line in fp_dataset:

            # getting the polarity of the sentence
            polarity = int(line[1])

            # obtaining the sentence, taking off the part '[X] XX - '
            line = ' '.join(((line.split())[3:]))

            # adding the unprocessed sentence to the set
            opinion_sets['raw'][['-', '+'][polarity]].append(line)

            # simplifying the sentence
            line = simplify_characters(line)

            # tokenize the sentence
            line = nltk.word_tokenize(line, language=PARAMETERS['LANGUAGE'])

            # removing stopwords from the sentence
            line = [word for word in line if (word not in nltk.corpus.stopwords.words(PARAMETERS['LANGUAGE']))]

            # adding the processed sentence to the set
            opinion_sets['processed'][['-', '+'][polarity]].append(line)

    # <----------------------------------------->
    # <--------> executing the methods <-------->
    # <----------------------------------------->

    if PARAMETERS['METHOD'] == 'CF':
        contrastiveness_first()
    elif PARAMETERS['METHOD'] == 'RF':
        representativeness_first()
    show_summary()
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
