#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
    Developed by: Otávio Augusto Ferreira Sousa
    Last modified: June 27, 2018

    This code evaluates a summary generated by COSummarizer.py which is based on the work of Kim & Zhai(2009).

    In order to execute this program, you should define some parameters. If you don't, default values will be used.
    All these parameters are predefined in a dictionary which is identified as 'PARAMETERS' and declared in the first section of this code.

    There are two ways to set these parameters.
    You could pass them as command line arguments or simply modify the dictionary called 'PARAMETERS' in this code.
    If you'd rather use command line arguments, here is an example on how you should do it:

    $ python Evaluation.py --dataset=sample_datasets/english/MicroMP3_design.txt --language=english --lambda=0.5 --summary=sample_outputs/output_eng.txt --human_labels_path=sample_human_labels/english/

    Please see details about the parameters below.

    DATASET  -> path and filename of the data set used to create the summary;
                each line of the data set must follow this format:
                [Polarity] Id - Sentence
                - Polarity: 0 if the sentence is negative, 1 if it is positive.
                - Id: id of the sentence. It goes from 1 to N for positive sentences and from 1 to M for negative ones.
                - Sentence: the opinionated sentence itself.
                If you want a few examples, please see the files in the folder 'sample_datasets/'.

    LANGUAGE -> it defines which language was considered during the generation of the summary;
                'portuguese' and 'english' are accepted.

    LAMBDA   -> a float value in range [0, 1] used in the summarization process;
                the bigger it is, the more importance you give to the representativeness of the summary;
                the smaller, the more importance you give to the contrastiveness of the summary.

    SUMMARY  -> path and filename of the generated summary itself;
                If you want a few examples, please see the files in the folder 'sample_outputs/'.

    HUMAN_LABELS_PATH -> directory that has the human labels generated for this data set;
                         for each data set, sentences are clustered into subtopics;
                         for each subtopic, sentence numbers marked in the data set file are listed;
                         each file in this directory must follow this format:

                         ------[data file name]
                         ---Pos
                         [subtopic1] [positiveSentenceNumber1] [positiveSentenceNumber2] ...
                         [subtopic2] [positiveSentenceNumber1] [positiveSentenceNumber2] ...
                         ...
                         ---Neg
                         [subtopic1] [negativeSentenceNumber1] [negativeSentenceNumber2] ...
                         [subtopic2] [negativeSentenceNumber1] [negativeSentenceNumber2] ...
                         ...

                         If you want a few examples, please see the files in the folder 'sample_human_labels/'.

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
PARAMETERS['SUMMARY'] = 'sample_outputs/output_ptbr.txt'
PARAMETERS['HUMAN_LABELS_PATH'] = 'sample_human_labels/portuguese/'


# <---------------------------->
# <-----> Common Modules <----->
# <---------------------------->


# basic functions and constants were used
import sys
import getopt
import os

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

# structures to store the human labeled clusters
human_labels = dict()


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


def find_contrastive_pairs_indices():
    """
    This function fills the structure contrastive_pairs according to the generated summary given in the file PARAMETERS['SUMMARY'].
    :return: none.
    """

    with open(PARAMETERS['SUMMARY'], encoding='utf-8') as fp_summary:
        cell_length = -1
        positive_sentence, negative_sentence = '', ''
        for line_id, line in enumerate(fp_summary):
            if line_id == 0:
                cell_length = max([len(x) for x in line.split('+')])
            elif 0 < line_id <= 2:
                continue
            elif line[0] == '+':
                positive_sentence = positive_sentence.strip()
                negative_sentence = negative_sentence.strip()
                positive_id, negative_id = -1, -1
                for sentence_id, sentence in enumerate(opinion_sets['raw']['+']):
                    if sentence == positive_sentence:
                        positive_id = sentence_id
                for sentence_id, sentence in enumerate(opinion_sets['raw']['-']):
                    if sentence == negative_sentence:
                        negative_id = sentence_id
                contrastive_pairs.append((positive_id, negative_id))
                positive_sentence, negative_sentence = '', ''
            else:
                positive_sentence += line[1:(cell_length + 1)]
                negative_sentence += line[(cell_length + 2):(2 * cell_length + 2)]


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


# <-------------------------------------->
# <-----> COS Evaluation Functions <----->
# <-------------------------------------->


def evaluate_representativeness():
    """
    The representativeness of a contrastive opinion summary S, denoted as r(S), measures how well the summary S represents the opinions expressed by the sentences in both X and Y .
    :return: [float] representativeness of the generated summary.
    """

    sum_in_x, sum_in_y = 0.0, 0.0

    for sX in range(len(opinion_sets['processed']['+'])):
        best_value = 0.0
        for (u, v) in contrastive_pairs:
            best_value = max(best_value, phi(sX, u, '+'))
        sum_in_x += best_value

    for sY in range(len(opinion_sets['processed']['-'])):
        best_value = 0.0
        for (u, v) in contrastive_pairs:
            best_value = max(best_value, phi(sY, v, '-'))
        sum_in_y += best_value

    return sum_in_x / len(opinion_sets['processed']['+']) + sum_in_y / len(opinion_sets['processed']['-'])


def evaluate_contrastiveness():
    """
    The contrastiveness of a contrastive opinion summary S, denoted as c(S), measures how well each u matches up with v in the summary.
    :return: [float] contrastiveness of the generated summary.
    """

    sumc = 0.0

    for (u, v) in contrastive_pairs:
        sumc += psi(u, v)

    return sumc / len(contrastive_pairs)


def evaluate_precision():
    """
    The precision of a summary with k contrastive sentence pairs is the percentage of the k pairs that are agreed by a human annotator. If a retrieved pair exists in an evaluator’s paired-cluster set, we assume that the pair is agreed by the annotator (i.e., “relevant”).
    :return: [float] precision of the generated summary.
    """

    count = 0

    for (u, v) in contrastive_pairs:
        pair_is_agreed = False

        for human_label in human_labels.values():
            for cluster in list(human_label.keys()):
                if len(human_label[
                           cluster].keys()) == 2:  # if the cluster is aligned, i.e., if it has the two polarities
                    if u in human_label[cluster]['Pos'] and v in human_label[cluster]['Neg']:
                        pair_is_agreed = True
                        break

        if pair_is_agreed:
            count += 1

    return float(count) / len(contrastive_pairs)


def evaluate_aspect_coverage():
    """
    The aspect coverage of a summary is the percentage of human-aligned clusters covered in the summary. If a pair of sentences appears in a human-aligned pair of clusters, we would assume that the aligned cluster is covered.
    :return: [float] aspect coverage of the generated summary.
    """

    covered_clusters = set()
    aligned_clusters = set()

    for (u, v) in contrastive_pairs:

        for human_label in human_labels.values():
            for cluster in list(human_label.keys()):
                if len(human_label[
                           cluster].keys()) == 2:  # if the cluster is aligned, i.e., if it has the two polarities
                    aligned_clusters.add(cluster)
                    if u in human_label[cluster]['Pos'] and v in human_label[cluster]['Neg']:
                        covered_clusters.add(cluster)

    return float(len(covered_clusters)) / len(aligned_clusters)


def evaluate_summary():
    """
    Function that prints the results in two perspectives: optimization results and NLP results.
    :return: none.
    """

    r_s = evaluate_representativeness()
    c_s = evaluate_contrastiveness()
    p = evaluate_precision()
    ac = evaluate_aspect_coverage()

    print(".::::::::::::::::::::::::::::.")
    print(".:: EVALUATING THE SUMMARY ::.")
    print(".::::::::::::::::::::::::::::.")
    print("###  OPTIMIZATION RESULTS  ###")
    print("|-> Contrastiveness:    %.2f |" % c_s)
    print("|-> Representativeness: %.2f |" % r_s)
    print("|-> Lambda parameter:   %.2f |" % PARAMETERS['LAMBDA'])
    print("|-> Optimal Value:      %.2f |" % (PARAMETERS['LAMBDA'] * r_s + (1.0 - PARAMETERS['LAMBDA']) * c_s))
    print("###      NLP RESULTS       ###")
    print("|-> Precision:          %.2f |" % p)
    print("|-> Aspect Coverage:    %.2f |" % ac)
    print("::::::::::::::::::::::::::::::")


def main(argv):
    global PARAMETERS, human_labels
    global contrastive_pairs, opinion_sets, phi_table, psi_table, omega_table

    # define the parameters according to what was given in the command line arguments;
    # if there are none, the program will consider the predefined parameters in the beginning of this code.
    try:
        opts, args = getopt.getopt(argv, "", ["dataset=", "language=", "lambda=", "summary=", "human_labels_path="])
        for opt, arg in opts:
            PARAMETERS[opt[2:].upper()] = arg
        PARAMETERS['LAMBDA'] = float(PARAMETERS['LAMBDA'])
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
    human_labels = dict()

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

    # <-------------------------------------------->
    # <--------> loading the human labels <-------->
    # <-------------------------------------------->

    # after loading the labels, each element in human_labels will be formatted as follows:
    #
    # human_labels[0] = {
    #       'speed': { 'Pos': [0, 2, 5], 'Neg': [1, 7] }
    #       'interface': { 'Pos': [0, 3, 6]}
    # }

    # loading labels from assessors
    for id_lbl, hl_filename in enumerate(os.listdir(PARAMETERS['HUMAN_LABELS_PATH'])):
        human_labels[id_lbl] = dict()
        with open(PARAMETERS['HUMAN_LABELS_PATH'] + hl_filename, encoding='utf-8') as fp_label:
            for line in fp_label:
                line = line.split('\n')[0]
                if line.find('------', 0, 6) == 0 and line[6:] == PARAMETERS['DATASET']:
                    polarity = ''
                    while True:
                        line = fp_label.readline().split('\n')[0]
                        if (line.find('------', 0, 6) == 0) or (len(line) == 0):
                            break
                        elif line.find('---', 0, 3) == 0:
                            polarity = line[3:]
                        else:
                            line = line.split()
                            if line[0] not in human_labels[id_lbl].keys():
                                human_labels[id_lbl][line[0]] = dict()
                            human_labels[id_lbl][line[0]][polarity] = [(int(x) - 1) for x in line[1:]]
                    break

    find_contrastive_pairs_indices()
    evaluate_summary()
    pass


if __name__ == "__main__":
    main(sys.argv[1:])