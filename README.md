# Transformer model for Seq2Seq Machine Translation

Transformer model for Chinese-English translation

## Basic Architecture

<p align="center">
<img src="https://github.com/P3n9W31/transformer-pytorch/blob/master/figures/architecture.png" width="300">
<img src="https://github.com/P3n9W31/transformer-pytorch/blob/master/figures/attn.png" width="200">
<img src="https://github.com/P3n9W31/transformer-pytorch/blob/master/figures/mh_attn.png" width="200">
</p>

> Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C] Advances in neural information processing systems. 2017: 5998-6008.

## Data Explanation

The Chinese-English translation data used in this project is just sample data, change them as you like.(Cn-863k, En-1.1M)

**Data Format**: 

```
sentence-1-word-1 sentence-1-word-2 sentence-1-word-3. [\n]
sentence-2-word-1 sentence-2-word-2 sentence-2-word-3 sentence-2-word-4. [\n]
......
```

Chinese-English data should be paired.



## Installation

Python3.6+ needed.

The following packages are needed:

```txt
regex==2018.1.10
terminaltables==3.1.0
torch==1.3.0
numpy==1.14.0
tensorboardX==1.9
```

Easily, you can install all requirement with:

```
pip3 install -r requirements.txt
```

## Usage

1. **Modifying hyperparameters**

   modify hyperparameters in hyperparams.py:

   ```
   +------------------+---------------------+
   | Parameters       | Value               |
   +------------------+---------------------+
   | source_train     | corpora/cn.txt      |
   | target_train     | corpora/en.txt      |
   | source_test      | corpora/cn.test.txt |
   | target_test      | corpora/en.test.txt |
   | batch_size       | 128                 |
   | batch_size_valid | 64                  |
   | lr               | 0.0002              |
   | logdir           | logdir              |
   | model_dir        | ./models/           |
   | maxlen           | 50                  |
   | min_cnt          | 0                   |
   | hidden_units     | 512                 |
   | num_blocks       | 12                  |
   | num_epochs       | 50                  |
   | num_heads        | 8                   |
   | dropout_rate     | 0.4                 |
   | sinusoid         | False               |
   | eval_epoch       | 1                   |
   | preload          | None                |
   | eval_script      | scripts/validate.sh |
   | check_frequence  | 10                  |
   +------------------+---------------------+
   ```

2. **Generating vocabulary**

   Generating vocabulary for training, run **prepro.py**:

3. **Training the model**

   Run **train.py**, start training model.

4. Visualize the training process on **tensorboard**

   ```bash
   tensorboard --logdir runs
   ```

   

## Evaluation

The evaluation metric for Chinese-English we use is case-insensitive BLEU. We use the `muti-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder) to compute the BLEU.

Result on tensorboard:

<p align="center">
<img src="https://github.com/P3n9W31/transformer-pytorch/blob/master/figures/result.jpg" width="700">
</p>

As the data is too simple, the results are **just a reference**.

## Device

Tested on CPU and Single GPU.

| Device Type | Device                                    | Speed                |
| ----------- | ----------------------------------------- | -------------------- |
| CPU         | Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz | 4 min 29 sec / Epoch |
| GPU         | GeForce GTX 1080 Ti                       | 48 sec / Epoch       |

## To Do

* Train on public dataset
* Test script

## License

MIT License