import sys
import argparse

from omegaconf import OmegaConf


def main(hp):
    train(hp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('env_name')
    args = parser.parse_args()

    if args.env_name == "cpu":
        args.config_name = "config_cpu"
        root_dir = "/home/mos/workshop/ProjectX"
    elif args.env_name == "gpu":
        args.config_name = "config_gpu"
        root_dir = "/home/mos/workshop/ProjectX"
    else:
        raise Exception("Sorry, there is no configuration for this device")

    config_dir = f"{root_dir}/configs"
    config_path = f"{config_dir}/{args.config_name}.yaml"
    hp = OmegaConf.load(config_path)
    hp.dir_setting.root_dir = root_dir

    sys.path.insert(0, root_dir)

    from src.training.train import train

    main(hp)
