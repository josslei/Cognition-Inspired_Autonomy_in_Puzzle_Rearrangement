import sys

import numpy as np
import matplotlib.pyplot as plt

from typing import List

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 visualize_loss.py <path_to_losses> [<graph_title>]')
        exit()
    
    path_losses: str = sys.argv[1]
    graph_tile: str = sys.argv[2] if len(sys.argv) > 2 else 'Loss Graph'

    # Read & prepare
    with open(path_losses, 'r') as fp:
        _losses = fp.readlines()
    losses: np.ndarray = np.asarray([float(l) for l in _losses])
    epochs: List[int] = [i for i, _ in enumerate(losses)]

    # Graph settings
    plt.title(graph_tile)
    # Axes settings
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, np.ceil(losses.max()))

    plt.plot(epochs, losses)
    plt.savefig('./loss_graph.png')
