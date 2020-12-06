import os
import time
import argparse

from hmm import HMM


def main(args):
    model = HMM(args.full_emissions, \
                args.add_one, \
                args.end_token, \
                args.config_path, \
                args.data_path, \
                args.save_model)

    model.test_model(args.data_path, args.save_test)

if __name__ == "__main__":
    """
    Program starts here.
    """

    # Parameters parsing
    parser = argparse.ArgumentParser(description="A POS tagger implementation based on supervised HMM.")
    parser.add_argument("-full_emissions", default=False, action='store_true', help="Flag to indicate should entries for not emissioned words by tags be added.")
    parser.add_argument("-add_one", default=False, action='store_true', help="Flag to indicate add-one smoothing usage.")
    parser.add_argument("-end_token", default=False, action='store_true', help="Flag to indicate usage of end token. Omit if don't want.")
    parser.add_argument("-config_path", default=None, type=str, help="Path to the directory where to store trained configs.")
    parser.add_argument("-data_path", default="./data", type=str, help="Path to the directory where the data for training/testing/eval are.")
    parser.add_argument("-save_model", default="./configs/config.json", type=str, help="Path to a file where to save trained config.")
    parser.add_argument("-save_test", default="./outputs/test.tt", type=str, help="Path to a file for test output.")
    args = parser.parse_args()

    # To measure runtime
    start_time = time.time()

    print("Program started...")
    
    main(args)
    
    print(f"Program ended. Runtime: {time.time() - start_time} sec")
