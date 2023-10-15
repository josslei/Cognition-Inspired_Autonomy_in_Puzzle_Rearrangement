import sys
import os

import torch
import tomli

from typing import Dict, List

from targf import TarGF_Tangram

TEST_BS: int = 9


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
    # Test
    error_mean: Dict[str, float] = { 'error_sum': 0.0, 'error_displacement': 0.0, 'error_rotation': 0.0 }
    for i in range(TEST_BS):
        err: List[Dict[str, float]]
        err = targf.test(config,
                         os.path.join(log_save_dir, f'score_net_epoch_{num_epochs - 1}.pt'),
                         os.path.join(log_save_dir, f'cnn_backbone_epoch_{num_epochs - 1}.pt'),
                         os.path.join(log_save_dir, f'process_visualization_{i}/'),
                         i)
        error_mean['error_sum'] += err[0]['error_sum']
        error_mean['error_displacement'] += err[0]['error_displacement']
        error_mean['error_rotation'] += err[0]['error_rotation']
    fp = open(os.path.join(log_save_dir, 'error_mean.txt'), 'w')
    for k in error_mean.keys():
        error_mean[k] /= TEST_BS
        print(f'{k}:{error_mean[k]}')
        fp.write(f'{k}:{error_mean[k]}\n')
    fp.close()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main()
