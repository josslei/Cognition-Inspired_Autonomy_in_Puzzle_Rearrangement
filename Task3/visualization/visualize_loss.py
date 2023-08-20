import sys

import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 visualize_loss.py <path_to_losses> [<graph_title>]')
        exit()
    
    path_losses: str = sys.argv[1]
    graph_tile: str = sys.argv[2] if len(sys.argv) > 2 else 'Loss Graph'

    plt.title(graph_tile)
    # Axes settings
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    with open(path_losses, 'r') as fp:
        losses = fp.readlines()
    losses = [float(l) for l in losses]
    epochs = [i for i, _ in enumerate(losses)]

    plt.plot(epochs, losses)
    plt.savefig('./loss_graph.png')
