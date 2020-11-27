import argparse

from hmm import HMM


def main():
    pass

if __name__ == "__main__":
    model = HMM(is_config=True, config_path="config.json")

    model.print_model()