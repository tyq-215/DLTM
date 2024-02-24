import load_dataset
from generator import Generator
from classfier import Classfier
from train import train
import torch
import selfies
import datetime

class Config():
    data_path = './data/mol/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    discriminator_method = 'Cond'
    load_pretrained_embed = False
    min_freq = 0
    max_length = 90
    embed_size = 256
    d_model = 256
    h = 8
    num_layers = 2
    batch_size = 2
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 5
    log_steps = 1
    eval_steps = 1000
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]
    slf_factor = 0.5
    # cyc_factor = 0.5
    cyc_factor = 0.5
    adv_factor = 10

    inp_drop_prob = 0
    epoches = 1


if __name__ == '__main__':

    # set_seed(20)
    config = Config()
    print(config.device)
    train_iters, test_iters, vocab, all_iter = load_dataset(config)
    print('Vocab size:', len(vocab))

    model_G = Generator(config, vocab).to(config.device)
    model_C = Classfier(config, vocab).to(config.device)

    train(config, vocab, model_G, model_C, train_iters, test_iters, all_iter)


