import copy
import time
import argparse
import random


def main(args):
    print("Welcome....")

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
    count_target_source = {key: copy.deepcopy(count_source_words) for key in count_target_source.keys()}
    probs_target_source = {key: copy.deepcopy({k: 1/target_voc_size for k in count_source_words.keys()}) for key in probs_target_source.keys()}

    #word = random.choice(list(count_source_words.keys()))
    #total_sum = sum([value[word] for value in probs_target_source.values()])
       
    # EM training

    # Choosing alignments


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