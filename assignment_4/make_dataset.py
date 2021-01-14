import argparse

"""
Helper script for transforming dataset in appropriate format for fast_align model
Format: foreign sentence ||| english sentence
"""

def make_dataset(args):
    source_path = args.data + '.' + args.f
    target_path = args.data + '.' + args.e
    dataset_path = args.data + ".fra-eng"

    with open(source_path, 'r', encoding="utf-8") as source_file, \
         open(target_path, 'r', encoding="utf-8") as target_file, \
         open(dataset_path, 'a', encoding="utf-8") as dataset_file:

        for (source_sent, target_sent) in zip(source_file, target_file):
            line = " ".join(source_sent.strip().split()) + " ||| " + " ".join(target_sent.strip().split()) + "\n"

            dataset_file.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to make dataset for fast_align.")
    parser.add_argument("-data", default="./data/hansards", type=str, help="Data filename path (default=./data/hansards)")
    parser.add_argument("-e", default="e", type=str, help="Suffix of Source lang filename (default=e)")
    parser.add_argument("-f", default="f", type=str, help="Suffix of Target lang filename (default=f)")
    args = parser.parse_args()

    make_dataset(args)
