import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from terminaltables import AsciiTable

from AttModel import AttModel
from bleu import bleu
from data_load import (
    get_batch_indices,
    load_cn_vocab,
    load_en_vocab,
    load_test_data,
    load_train_data,
)
from hyperparams import Hyperparams as hp
from util import get_logger

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# log
if not os.path.exists("log"):
    os.mkdir("log")

log_path = os.path.join(
    "log", "log-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".txt"
)
logger = get_logger(log_path)


# validation script
def bleu_script(f):
    ref_stem = hp.target_test
    cmd = "{eval_script} {refs} {hyp}".format(
        eval_script=hp.eval_script, refs=ref_stem, hyp=f
    )
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err)
        sys.exit(1)
    bleu = float(out)
    return bleu


def train():
    paras = [["Parameters", "Value"]]
    for key, value in hp.__dict__.items():
        if "__" not in key:
            paras.append([str(key), str(value)])
    paras_table = AsciiTable(paras)
    logger.info("\n" + str(paras_table.table))
    score_list = [
        [
            "epoch_multi_bleu",
            "epoch_bleu_1_gram",
            "epoch_bleu_2_gram",
            "epoch_bleu_3_gram",
            "epoch_bleu_4_gram",
            "epoch",
        ]
    ]

    global_batches = 0
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(cn2idx)
    dec_voc = len(en2idx)
    writer = SummaryWriter()
    # Load data
    X, Y = load_train_data()
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    model = AttModel(hp, enc_voc, dec_voc)
    model.train()
    model.to(device)
    torch.backends.cudnn.benchmark = True  # may speed up Forward propagation
    if not os.path.exists(hp.model_dir):
        os.makedirs(hp.model_dir)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    for epoch in range(1, hp.num_epochs + 1):
        current_batches = 0
        for index, current_index in get_batch_indices(len(X), hp.batch_size):
            x_batch = torch.LongTensor(X[index]).to(device)
            y_batch = torch.LongTensor(Y[index]).to(device)

            optimizer.zero_grad()
            loss, _, acc = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()

            global_batches += 1
            current_batches += 1
            if current_batches % 1 == 0:
                writer.add_scalar(
                    "./loss",
                    scalar_value=loss.detach().cpu().numpy(),
                    global_step=global_batches,
                )
                writer.add_scalar(
                    "./acc",
                    scalar_value=acc.detach().cpu().numpy(),
                    global_step=global_batches,
                )

            if (
                current_batches % 10 == 0
                or current_batches == 0
                or current_batches == num_batch
            ):
                logger.info(
                    "Epoch: {} batch: {}/{}({:.2%}), loss: {:.6}, acc: {:.4}".format(
                        epoch,
                        current_batches,
                        num_batch,
                        current_batches / num_batch,
                        loss.data.item(),
                        acc.data.item(),
                    )
                )

        if epoch % hp.check_frequency == 0 or epoch == hp.num_epochs:
            checkpoint_path = hp.model_dir + "/model_epoch_%02d" % epoch + ".pth"
            torch.save(model.state_dict(), checkpoint_path)

        # eval
        score_list = evaluate(model, epoch, writer, score_list)
    writer.close()
    score_table = AsciiTable(score_list)
    logger.info("\n" + score_table.table)


def evaluate(model, epoch, writer, score_list):
    # Load data
    X, Sources, Targets = load_test_data()
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    model.eval()
    model.to(device)
    # Inference
    if not os.path.exists("results"):
        os.mkdir("results")
    list_of_refs = []
    hypotheses = []
    assert hp.batch_size_valid <= len(
        X
    ), "test batch size is large than total data length. Check your data or change batch size."

    for i in range(len(X) // hp.batch_size_valid):
        # Get mini-batches
        x = X[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]
        sources = Sources[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]
        targets = Targets[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]

        # Autoregressive inference
        x_ = torch.LongTensor(x).to(device)
        preds_t = torch.LongTensor(
            np.zeros((hp.batch_size_valid, hp.maxlen), np.int32)
        ).to(device)
        preds = preds_t
        _, _preds, _ = model(x_, preds)
        preds = _preds.data.cpu().numpy()

        # prepare data for BLEU score
        for source, target, pred in zip(sources, targets, preds):
            got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
            ref = target.split()
            hypothesis = got.split()
            if len(ref) > 3 and len(hypothesis) > 3:
                list_of_refs.append([ref])
                hypotheses.append(hypothesis)

    ix = np.random.randint(0, hp.batch_size_valid)
    sampling_result = []
    sampling_result.append(["Key", "Value"])
    sampling_result.append(["Source", " ".join(idx2cn[idx] for idx in X[ix]).split("</S>")[0].strip()])
    sampling_result.append(["Target", Targets[ix]])
    sampling_result.append(["Predict", " ".join(idx2en[idx] for idx in preds[ix]).split("</S>")[0].strip()])
    sampling_table = AsciiTable(sampling_result)
    logger.info("===========sampling START===========")
    logger.info("\n" + str(sampling_table.table))
    logger.info("===========sampling DONE===========")
    # Calculate BLEU score
    hypotheses = [" ".join(x) for x in hypotheses]

    p_tmp = tempfile.mktemp()
    f_tmp = open(p_tmp, "w")
    f_tmp.write("\n".join(hypotheses))
    f_tmp.close()
    multi_bleu = bleu_script(p_tmp)
    bleu_1_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=1)
    bleu_2_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=2)
    bleu_3_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=3)
    bleu_4_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=4)

    writer.add_scalar("./bleu_1_gram", bleu_1_gram, epoch)
    writer.add_scalar("./bleu_2_gram", bleu_2_gram, epoch)
    writer.add_scalar("./bleu_3_gram", bleu_3_gram, epoch)
    writer.add_scalar("./bleu_4_gram", bleu_4_gram, epoch)
    writer.add_scalar("./multi-bleu", multi_bleu, epoch)

    bleu_result = [
        ["multi-bleu", "bleu_1-gram", "bleu_2-gram", "bleu_3-gram", "bleu_4-gram",],
        [multi_bleu, bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram,],
    ]
    bleu_table = AsciiTable(bleu_result)
    logger.info("BLEU score for Epoch-{}: ".format(epoch) + "\n" + bleu_table.table)
    score_list.append(
        [multi_bleu, bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram, epoch,]
    )

    return score_list


if __name__ == "__main__":
    train()
