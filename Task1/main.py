from network.sde import marginal_prob_std
from targf import Dataset_KILOGRAM, TarGF_Tangram_Ball

def main():
    targf = TarGF_Tangram_Ball(sigma=25,
                               path_kilogram_dataset='./kilogram/parsed.json',
                               is_json=True,
                               learning_rate=0.002,
                               batch_size=127,
                               num_epochs=10000)
    targf.train()
    targf.test('logs/score_net_epoch_9999.pt', 'logs/process_visualization_0/', 0, [2, 3, 4, 5, 6])
    pass

if __name__ == '__main__':
    main()
