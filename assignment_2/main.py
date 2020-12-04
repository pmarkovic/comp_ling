import os
import argparse

from hmm import HMM


def main():
    pass

if __name__ == "__main__":

    model = HMM(True, \
                False, \
                "crude", \
                False, \
                config_path=None, \
                train_path="data", \
                save_model_path="./configs/config_7.json", \
                save_test_path="./tests/test7.tt")
    
    model.test_model("data")
    #model.print_model()
    #model.check_total_probs()

    #model.do_viterbi(["2", "1", "1"])
    #model.print_tags()
