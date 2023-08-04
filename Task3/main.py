import sys
import os

import tomli

from targf import TarGF_Tangram


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 main.py <path_to_config>')
        exit()

    path_config: str = sys.argv[1]

    with open(path_config, 'rb') as fp:
        config = tomli.load(fp)
    log_save_dir: str = f'./logs/log_{config["config_name"]}'

    targf = TarGF_Tangram(config=config)

    num_epochs: int = config['training']['num_epochs']
    targf.train(config=config, log_save_dir=log_save_dir)
    for i in range(20):
        targf.test(config,
                   os.path.join(log_save_dir, f'score_net_epoch_{num_epochs - 1}.pt'),
                   os.path.join(log_save_dir, f'process_visualization_{i}/'),
                   i)


if __name__ == '__main__':
    main()
