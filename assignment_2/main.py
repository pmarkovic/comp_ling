import os
import argparse

from hmm import HMM


def main():
    pass

if __name__ == "__main__":

    model = HMM(True, \
                config_path="./configs/config_1.json", \
                train_path=None, \
                save_model_path="./configs/config_1.json", \
                save_test_path=None)
    
    #model.test_model("data")
    model.print_model()
    #model.check_total_probs()

    #model.do_viterbi(["2", "1", "1"])
    #model.print_tags()
