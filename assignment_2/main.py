import os
import argparse

from hmm import HMM


def main():
    pass

if __name__ == "__main__":
    model = HMM(is_config=False, config_path="default.json", train_path=os.path.join(os.getcwd(), "data"))
    #model.print_model()

    #model.do_viterbi(["2", "1", "1"])
    #model.print_tags()
