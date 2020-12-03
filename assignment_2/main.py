import os
import argparse

from hmm import HMM


def main():
    pass

if __name__ == "__main__":
    model = HMM(2, is_config=True, config_path="./configs/config_0.json", train_path=os.path.join(os.getcwd(), "data"), save_model=True, save_path=os.path.join(os.getcwd(), "configs", "config_0.json"))
    
    model.test_model("data")
    #model.check_total_probs()

    #model.do_viterbi(["2", "1", "1"])
    #model.print_tags()
