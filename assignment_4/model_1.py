import sys
import copy
import time
import argparse
import random


def main(args):
    # Initialization
    source_path = args.data + '.' + args.e
    target_path = args.data + '.' + args.f

    count_source_words = dict()
    count_target_source = dict()
    probs_target_source = dict()
    bitext = list()

    with open(source_path, 'r', encoding="utf-8") as source_file, open(target_path, 'r', encoding="utf-8") as target_file:
        for pair, (source_sent, target_sent) in enumerate(zip(source_file, target_file)):
            source_sent_words = source_sent.strip().split()
            target_sent_words = target_sent.strip().split()
            bitext.append([source_sent_words, target_sent_words])

            for word in source_sent_words:
                if word not in count_source_words:
                    count_source_words[word] = 0

            for word in target_sent_words:
                if word not in count_target_source:
                    count_target_source[word] = 0
                    probs_target_source[word] = 0

            if pair == args.n-1:
                break

    target_voc_size = len(probs_target_source)
    probs_target_source = {key: copy.deepcopy({k: 1/target_voc_size for k in count_source_words.keys()}) for key in probs_target_source.keys()}

    #word = random.choice(list(count_source_words.keys()))
    #total_sum = sum([value[word] for value in probs_target_source.values()])
       
    # EM training
    for iter in range(args.i):
        # All counts to zero
        count_source_words = {key: 0 for key in count_source_words.keys()}
        count_target_source = {key: copy.deepcopy(count_source_words) for key in count_target_source.keys()}

        # E-step
        for example, (source_sent, target_sent) in enumerate(bitext):
            for target_word in target_sent:
                # Calculate value for normalization term
                norm_term = sum([probs_target_source[target_word][source_word] for source_word in source_sent])

                for source_word in source_sent:
                    # Calculate expected count
                    expected_count = probs_target_source[target_word][source_word] / norm_term
                    count_target_source[target_word][source_word] += expected_count
                    count_source_words[source_word] += expected_count

        # M-step
        for target_word in probs_target_source.keys():
            for source_word in count_source_words.keys():
                probs_target_source[target_word][source_word] = count_target_source[target_word][source_word] / count_source_words[source_word]

    # Choosing alignments
    for example, (source_sent, target_sent) in enumerate(bitext):
        alignment = list()

        for target_pos, target_word in enumerate(target_sent):
            best_prob = 0
            best_position = 0

            for position, source_word in enumerate(source_sent):
                if probs_target_source[target_word][source_word] > best_prob:
                    best_prob = probs_target_source[target_word][source_word]
                    best_position = position

            sys.stdout.write("%i-%i " % (target_pos, best_position))
        sys.stdout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An implementation of IBM Model 1 for word alignments.")
    parser.add_argument("-data", default="./data/hansards", type=str, help="Data filename path (default=./data/hansards)")
    parser.add_argument("-e", default="e", type=str, help="Suffix of Source lang filename (default=e)")
    parser.add_argument("-f", default="f", type=str, help="Suffix of Target lang filename (default=f)")
    parser.add_argument("-i", default=10, type=int, help="Number of iteration for EM algorithm (default=10)")
    parser.add_argument("-n", default=100000000000, type=int, help="Number of sentences to use for training and alignment")
    #parser.add_argument("-draw", default=False, action='store_true', help="Flag to indicate if parse trees should be print or draw. Omit if don't want.")
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time() - start_time