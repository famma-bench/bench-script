import argparse
from omegaconf import OmegaConf
from famma_runner.runners import Runner

if __name__ == "__main__":
    """
    analyze the scores of the answers generated by a model and save the results to files.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, default="../configs/analyze_config.yaml",
                        help="The dir of evaluation config file.")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_dir)

    runner = Runner.build_from_config(config)

    runner.run()