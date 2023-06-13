from network.sde import marginal_prob_std
from targf import Dataset_KILOGRAM, TarGF_Tangram_Ball

def main():
    #dataset = Dataset_KILOGRAM('./kilogram/dataset/')
    # TODO: Ask what is `sigma`
    targf = TarGF_Tangram_Ball(sigma=25,
                               path_kilogram_dataset='./kilogram/parsed.json',
                               is_json=True,
                               learning_rate=0.0002,
                               batch_size=127,
                               num_epochs=10000)
    targf.train()
    pass

if __name__ == '__main__':
    main()
