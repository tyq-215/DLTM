import os
import random
import time
import torch
import numpy as np
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from rdkit.Chem import QED
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop
from math import log
import pandas as pd
from utils.utils import batch_preprocess, get_lengths, choose_best_smiles


def train_c_step(config, vocab, model_G, model_C, optimizer_C, batch, temperature):
    model_G.eval()
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_labels = batch_preprocess(batch, pad_idx, eos_idx)
    rev_labels = 1 - raw_labels
    batch_size = inp_tokens.size(0)
    input_0 = batch[0]
    input_1 = batch[1]
    smiles1 = tensor2text(vocab, input_0)
    smiles2 = tensor2text(vocab, input_1)
    fp_str = []
    for smile in smiles1:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
        fp_str.append(fp)
    for smile in smiles2:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
        fp_str.append(fp)
    real_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in fp_str]).to(device)
    real_fp = real_fp.detach()
    with torch.no_grad():
        raw_gen_log_probs = model_G(
            inp_tokens,
            None,
            inp_lengths,
            raw_labels,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
            fp=real_fp,
        )
        rev_gen_log_probs = model_G(
            inp_tokens,
            None,
            inp_lengths,
            rev_labels,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
            fp=real_fp,
        )

    raw_gen_soft_tokens = raw_gen_log_probs.exp()
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)

    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

    raw_gold_log_probs = model_C(inp_tokens, inp_lengths, raw_labels)
    rev_gold_log_probs = model_C(inp_tokens, inp_lengths, rev_labels)
    gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
    raw_gold_labels = torch.ones_like(raw_labels)
    raw_gold_labels = raw_gold_labels - random.uniform(0, 0.2)
    rev_gold_labels = torch.zeros_like(rev_labels)
    gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)

    raw_gen_log_probs = model_C(raw_gen_soft_tokens, raw_gen_lengths, raw_labels)
    rev_gen_log_probs = model_C(rev_gen_soft_tokens, rev_gen_lengths, rev_labels)
    gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
    raw_gen_labels = torch.ones_like(raw_labels)
    rev_gen_labels = torch.zeros_like(rev_labels)
    gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)

    cls_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
    cls_labels = torch.cat((gold_labels.long(), gen_labels.long()), 0)
    cls_loss = loss_fn(cls_log_probs, cls_labels)
    assert len(cls_loss.size()) == 1
    cls_loss = cls_loss.sum() / batch_size
    loss = cls_loss

    optimizer_C.zero_grad()
    loss.backward()
    clip_grad_norm_(model_C.parameters(), 5)
    optimizer_C.step()

    return cls_loss.item()


def f_step(config, vocab, model_G, model_C, optimizer_F, batch, temperature, drop_decay,
           cyc_rec_enable=True):
    model_C.eval()
    input_0 = batch[0]
    input_1 = batch[1]
    smiles1 = tensor2text(vocab, input_0)
    smiles2 = tensor2text(vocab, input_1)
    fp_str = []
    for smile in smiles1:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
        fp_str.append(fp)
    for smile in smiles2:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
        fp_str.append(fp)
    real_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in fp_str]).to(device)
    real_fp = real_fp.detach()
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_labels = batch_preprocess(batch, pad_idx, eos_idx)
    rev_labels = 1 - raw_labels
    batch_size = inp_tokens.size(0)
 
    token_mask = (inp_tokens != pad_idx).float()

    optimizer_F.zero_grad()

    # self reconstruction loss

    noise_inp_tokens = word_drop(
        inp_tokens,
        inp_lengths,
        config.inp_drop_prob * drop_decay,
        vocab
    )
    noise_inp_lengths = get_lengths(noise_inp_tokens, eos_idx)

    slf_log_probs = model_G(
        noise_inp_tokens,
        inp_tokens,
        noise_inp_lengths,
        raw_labels,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
        fp=real_fp,

    )
    a = slf_log_probs.transhighe(1, 2)
    slf_rec_loss = loss_fn(slf_log_probs.transhighe(1, 2), inp_tokens) * token_mask
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor

    # cycle consistency loss
    if not cyc_rec_enable:

        slf_rec_loss.backward()
        optimizer_F.step()
        model_C.train()
        return slf_rec_loss.item(), 0, 0

    gen_log_probs = model_G(
        inp_tokens,
        None,
        inp_lengths,
        rev_labels,
        generate=True,
        differentiable_decode=True,
        temperature=temperature,
        fp=real_fp,

    )
    # identity_loss
    input_token = inp_tokens
    gap_size = gen_log_probs.shape[1] - input_token.shape[1]
    gap_tensor = torch.zeros([input_token.shape[0], gap_size]).cuda()
    input_token = torch.cat((input_token, gap_tensor), dim=1).type(torch.LongTensor).cuda()
    token_mask_ien = token_mask
    token_mask_ien = torch.cat((token_mask_ien, gap_tensor), dim=1)
    identity_loss = loss_fn(gen_log_probs.transhighe(1, 2), input_token) * token_mask_ien
    identity_loss = config.iden_factor * identity_loss.sum() / batch_size

    gen_soft_tokens = gen_log_probs.exp()
    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

    cyc_log_probs = model_G(
        gen_soft_tokens,
        inp_tokens,
        gen_lengths,
        raw_labels,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
        fp=real_fp,
    )

    cyc_rec_loss = loss_fn(cyc_log_probs.transhighe(1, 2), inp_tokens) * token_mask
    cyc_rec_loss = cyc_rec_loss.sum() / batch_size
    cyc_rec_loss *= config.cyc_factor

    cls_log_porbs = model_C(gen_soft_tokens, gen_lengths, rev_labels)
    cls_labels = torch.ones_like(rev_labels)
    cls_loss = loss_fn(cls_log_porbs, cls_labels)
    cls_loss = cls_loss.sum() / batch_size
    cls_loss *= config.cls_factor

    (slf_rec_loss + cls_loss + cyc_rec_loss + identity_loss).backward()

    # update parameters

    clip_grad_norm_(model_G.parameters(), 5)
    optimizer_F.step()

    model_C.train()

    return slf_rec_loss.item(), cyc_rec_loss.item(), cls_loss.item(), identity_loss.item()


def train(config, vocab, model_G, model_C, train_iters, test_iters, all_iters):
    optimizer_G = optim.Adam(model_G.parameters(), lr=config.lr_F, weight_decay=config.L2)
    optimizer_C = optim.Adam(model_C.parameters(), lr=config.lr_D, weight_decay=config.L2)

    his_d_cls_loss = []
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_cls_loss = []
    his_f_identity_loss = []

    # writer = SummaryWriter(config.log_dir)
    global_step = 0
    model_G.train()
    model_C.train()
    config.save_folder = config.save_path + '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
    os.makedirs(config.save_folder)
    os.makedirs(config.save_folder + '/ckpts')
    print('Save Path:', config.save_folder)
    print('Model G pretraining......')

    for i, batch in enumerate(train_iters):
        if i >= config.F_pretrain_iter:
            break
        slf_loss, cyc_loss, _ = f_step(config, vocab, model_G, model_C, optimizer_G, batch,  temperature, drop_decay, False)
        his_f_slf_loss.append(slf_loss)
        his_f_cyc_loss.append(cyc_loss)

        if (i + 1) % 10 == 0:
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            his_f_slf_loss = []
            his_f_cyc_loss = []
            print('[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f}'
                  .format(i + 1, avrg_f_slf_loss, avrg_f_cyc_loss))

    print('Training start......')

    def calc_temperature(temperature_config, step):
        num = len(temperature_config)
        for i in range(num):
            t_a, s_a = temperature_config[i]
            if i == num - 1:
                return t_a
            t_b, s_b = temperature_config[i + 1]
            if s_a <= step < s_b:
                k = (step - s_a) / (s_b - s_a)
                temperature = (1 - k) * t_a + k * t_b
                return temperature

    batch_iters = iter(train_iters)
    while train_iters.high_iter.epoch < config.epoches:
        drop_decay = calc_temperature(config.drop_rate_config, global_step)
        temperature = calc_temperature(config.temperature_config, global_step)

        for _ in range(config.iter_D):
            batch = next(batch_iters)
            d_cls_loss = train_c_step(
                config, vocab, model_G, model_C, optimizer_C, batch, temperature
            )
            his_d_cls_loss.append(d_cls_loss)

        for _ in range(config.iter_F):
            batch = next(batch_iters)
            f_slf_loss, f_cyc_loss, f_cls_loss, f_indentity_loss = f_step(
                config, vocab, model_G, model_C, optimizer_G, batch, temperature, drop_decay
            )
            his_f_slf_loss.append(f_slf_loss)
            his_f_cyc_loss.append(f_cyc_loss)
            his_f_cls_loss.append(f_cls_loss)
            his_f_identity_loss.append(f_indentity_loss)
        global_step += 1

        if global_step % config.log_steps == 0:
            avrg_d_cls_loss = np.mean(his_d_cls_loss)
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            avrg_f_cls_loss = np.mean(his_f_cls_loss)
            avg_f_identity_loss = np.mean(his_f_identity_loss)
            log_str = '[iter {} epoch{}] d_cls_loss: {:.4f}  ' + \
                      'f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  ' + \
                      'f_cls_loss: {:.4f}  f_iden_loss: {:.4f}  drop: {:.4f}'
            print(log_str.format(
                global_step, train_iters.high_iter.epoch, avrg_d_cls_loss,
                avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_cls_loss,
                avg_f_identity_loss, config.inp_drop_prob * drop_decay
            ))

            torch.save(model_G.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_F.pth')
            torch.save(model_C.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_D.pth')
            gold_text, raw_output, rev_output = auto_eval(config, vocab, model_G, test_iters, global_step, temperature,train_iters.high_iter.epoch)
            with open(''.format(global_step), 'a+', encoding='utf-8') as f:
                for data in rev_output:
                    f.write(data + '\n')
                f.close()


def auto_eval(config, vocab, model_G, test_iters, global_step, temperature, epoch):
    model_G.eval()
    vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']

    def inference(data_iter, raw_label):
        original_smiles = []
        raw_output = []
        rev_output = []
        sum_rev_output = []
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_labels = torch.full_like(inp_tokens[:, 0], raw_label)
            rev_labels = 1 - raw_labels
            smiles = tensor2text(vocab, inp_tokens)
            fp_str = []
            smiles2 = []
            for smile in smiles:
                mol = Chem.MolFromSmiles(smile)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
                fp_str.append(fp)
            real_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in fp_str]).to(device)
            real_fp = real_fp.detach()
            with torch.no_grad():
                raw_log_probs = model_G(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_labels,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                    fp=real_fp,
                )
            with torch.no_grad():
                rev_log_probs = model_G(
                    inp_tokens,
                    None,
                    inp_lengths,
                    rev_labels,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                    fp=real_fp,
                    test=True,
                )

            original_smiles += tensor2text(vocab, inp_tokens.cpu())
            tensor = rev_log_probs
            smiles_arr = tensor2text(vocab, tensor.argmax(-1).cpu())
            # chose a best smile from generated smiles
            generated_smiles = choose_best_smiles(smiles_arr,original_smiles)
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
            rev_output += generated_smiles
            df = pd.DataFrame(sum_rev_output)
            df.to_csv("./outputs/mol/low2high_sum({}).txt".format(epoch), index=False)

        return gold_text, raw_output, rev_output

    low_iter = test_iters.low_iter
    gold_text, raw_output, rev_output = inference(low_iter, 0)
    return gold_text, raw_output, rev_output
