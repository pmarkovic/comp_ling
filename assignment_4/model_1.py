import time
import argparse


def main(args):
    print("Welcome....")

    print(f"Data: {args.data}")
    print(f"Source: {args.e}")
    print(f"Target: {args.f}")
    print(f"Iterations: {args.i}")
    print(f"Num of sentence pairs: {args.n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An implementation of IBM Model 1 for word alignments.")
    parser.add_argument("-data", default="./data/hansards", type=str, help="Data filename prefix (default=data)")
    parser.add_argument("-e", default="e", type=str, help="Suffix of English filename (default=e)")
    parser.add_argument("-f", default="f", type=str, help="Suffix of French filename (default=f)")
    parser.add_argument("-i", default=10, type=int, help="Number of iteration for EM algorithm (default=10)")
    parser.add_argument("-n", default=100000000000, type=int, help="Number of sentences to use for training and alignment")
    #parser.add_argument("-draw", default=False, action='store_true', help="Flag to indicate if parse trees should be print or draw. Omit if don't want.")
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time() - start_time