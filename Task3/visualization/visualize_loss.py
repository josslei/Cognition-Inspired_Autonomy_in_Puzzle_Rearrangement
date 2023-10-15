import sys

import numpy as np
import matplotlib.pyplot as plt
import math

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
    losses_each: np.ndarray = np.asarray([float(l) for l in _losses])
    losses = []
    avg = 0
    count = 0
    for l in losses_each:
        avg += l
        count += 1
        if count == 100:
            losses += [avg / count]
            avg = 0
            count = 0
    losses: np.ndarray = np.array(losses)

    epochs: List[int] = [i for i, _ in enumerate(losses)]

    # Graph settings
    plt.title(graph_tile)
    # Axes settings
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.ylim(0, np.ceil(losses.max()))
    plt.ylim(0, 20)

    plt.plot(epochs, losses)
    plt.savefig('./loss_graph.png')
