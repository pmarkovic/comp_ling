import sys
import time
import argparse
from collections import defaultdict


"""
Implementation of IBM Model 1 for word alignment for statistical machine translation task
"""


def main(args):
    # Initialization
    # Source are foreign examples, target are english examples
    source_path = args.data + '.' + args.f
    target_path = args.data + '.' + args.e
    threshold = args.t

    # Helper dictionaries to transform sentences to encodings
    source_count = 0
    target_count = 0
    source2ind = dict()
    target2ind = dict()
    bitext = list()

    with open(source_path, 'r', encoding="utf-8") as source_file, open(target_path, 'r', encoding="utf-8") as target_file:
        for pair, (source_sent, target_sent) in enumerate(zip(source_file, target_file)):
            source_sent_words = list()
            target_sent_words = list()

            # Transform source sentence into appropriate encoding
            for source_word in source_sent.strip().split():
                if source_word not in source2ind:
                    source2ind[source_word] = source_count
                    source_count += 1
                source_sent_words.append(source2ind[source_word])

            # Transform target sentence into appropriate encoding
            for target_word in target_sent.strip().split():
                if target_word not in target2ind:
                    target2ind[target_word] = target_count
                    target_count += 1
                target_sent_words.append(target2ind[target_word])

            # Store encoded example pair
            bitext.append([source_sent_words, target_sent_words])
                    
            if pair == args.n-1:
                break
    
    # Initialize P(e|f)
    probs_target_source = defaultdict(lambda: 1/source_count)
       
    # EM training
    em_start = time.time()
    for iter in range(args.i):
        # All counts to zero
        count_source_words = defaultdict(float)
        count_target_source = defaultdict(float)
        
        # E-step
        for example, (source_sent, target_sent) in enumerate(bitext):
            for target_word in target_sent:
                # Calculate value for normalization term
                norm_term = sum([probs_target_source[(target_word, source_word)] for source_word in source_sent])

                for source_word in source_sent:
                    # Calculate expected counts
                    expected_count = probs_target_source[(target_word, source_word)] / norm_term
                    count_target_source[(target_word, source_word)] += expected_count
                    count_source_words[source_word] += expected_count

        # M-step
        # key is (e, f)
        probs_target_source.update({key: count_target_source[key]/count_source_words[key[1]] for key in probs_target_source.keys()})

    em_end = time.time()

    # Choosing alignments
    alignment_start = time.time()
    for example, (source_sent, target_sent) in enumerate(bitext):
        for target_pos, target_word in enumerate(target_sent):
            for position, source_word in enumerate(source_sent):
                if probs_target_source[(target_word, source_word)] > threshold:
                    sys.stdout.write("%i-%i " % (position, target_pos))
        sys.stdout.write("\n")

    alignment_end = time.time()
    #print(f"EM training time: {em_end - em_start}")
    #print(f"Alignment time: {alignment_end - alignment_start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An implementation of IBM Model 1 for word alignments.")
    parser.add_argument("-data", default="./data/hansards", type=str, help="Data filename path (default=./data/hansards)")
    parser.add_argument("-e", default="e", type=str, help="Suffix of Source lang filename (default=e)")
    parser.add_argument("-f", default="f", type=str, help="Suffix of Target lang filename (default=f)")
    parser.add_argument("-i", default=10, type=int, help="Number of iteration for EM algorithm (default=10)")
    parser.add_argument("-t", default=0.5, type=float, help="Threshold for alignment.")
    parser.add_argument("-n", default=100000000000, type=int, help="Number of sentences to use for training and alignment")
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    #print(f"Total time: {time.time() - start_time}")