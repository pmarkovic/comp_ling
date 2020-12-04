import os
import argparse

from hmm import HMM


def main():
    pass

if __name__ == "__main__":

    model = HMM(False, \
                "crude", \
                True, \
                config_path=None, \
                train_path="data", \
                save_model_path="./configs/config_3.json", \
                save_test_path="./tests/test3.tt")
    
    model.test_model("data")
    #model.print_model()
    #model.check_total_probs()

    #model.do_viterbi(["2", "1", "1"])
    #model.print_tags()
