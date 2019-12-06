# Transformer model for Seq2Seq Machine Translation

Transformer model for Chinese-English translation

## Basic architecture

<p align="center">
<img src="https://github.com/P3n9W31/transformer-pytorch/blob/master/figures/architecture.png" width="300">
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

## Results on Chinese-English translation

1. Epoch-1

   ```
   +---------+--------------------------------------------------------------------------------------------------------------------------------------------------+
   | Key     | Value                                                                                                                                            |
   +---------+--------------------------------------------------------------------------------------------------------------------------------------------------+
   | Source  | 海南 北 大 青鸟 软件 有限 公司 副 总经理 安金龙 教授 说 听到 这个 事件 发生 后 我们 感到 十分 气愤                                                         |
   | Target  | professor an jinlong  deputy manager of the hainan beida qingdao software company  said  we felt extremely angry after hearing of this incident  |
   | Predict | the the the the the of the the the the the the the the the the the the and and to and of                                                         |
   +---------+--------------------------------------------------------------------------------------------------------------------------------------------------+
   ```

2. Epoch-10

   ```
   +---------+----------------------------------------------------------------------------------------------------------------------------------+
   | Key     | Value                                                                                                                            |
   +---------+----------------------------------------------------------------------------------------------------------------------------------+
   | Source  | 嘉木样 是 在 这里 召开 的 中国 宗教界 世界 和平 问题 研讨会 上 说 这 番 话 的                                                            |
   | Target  | jamyang made these remarks at  china 's religious circles ' symposium on the question of world peace   which is being held here  |
   | Predict | without with air air with with at with in military or and and and and and and and and of and and                                 |
   +---------+----------------------------------------------------------------------------------------------------------------------------------+
   ```

3. Epoch-20

   ```
   +---------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Key     | Value                                                                                                                                                                 |
   +---------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Source  | 经济 全球化 的 内容 主要 体现 在 国际 贸易 国际 金融 国际 生产 三 大 基本 领域                                                                                                |
   | Target  | what is involved in economic globalization is mainly to be seen in the three basic areas of international trade  international finance  and international production  |
   | Predict | for for provide for for for for for for global percent for at that and and and and and for and and                                                                    |
   +---------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   ```

4. Epoch-30

   ```
   +---------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Key     | Value                                                                                                                                                                  |
   +---------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Source  | 博什格拉夫 没有 说 但 旁观者清 就 是 因为 美国 以为 自己 无所不在 无所不能 要求 所有 国家 唯它 的 马首是瞻                                                                        |
   | Target  | as a bystander always sees things more clearly  it is because the united states sees itself omnipresent and omnipotent and demands other countries to follow its lead  |
   | Predict | we the have we we have the the the an it the and we and and and and and of and                                                                                         |
   +---------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   ```

5. Epoch-40

   ```
   +---------+------------------------------------------------------------------------------------------------------------------------------------------+
   | Key     | Value                                                                                                                                    |
   +---------+------------------------------------------------------------------------------------------------------------------------------------------+
   | Source  | 目前 台湾 正 处在 十字路口 走向 分裂 战争 还是 走向 统一 和平 台湾 面临 着 重大 的 抉择                                                           |
   | Target  | taiwan is now at a crossroad  taiwan is faced with a major choice  moving towards separation and war or towards reunification and peace  |
   | Predict | redistribution with rich in rich for about the the military military of and by and and and and and of of and                             |
   +---------+------------------------------------------------------------------------------------------------------------------------------------------+
   ```

6. Epoch-50

   ```
   +---------+----------------------------------------------------------------------------------------------------------------------------------------------+
   | Key     | Value                                                                                                                                        |
   +---------+----------------------------------------------------------------------------------------------------------------------------------------------+
   | Source  | 不论 目的 何在 理由 多么 冠冕堂皇 战争 都 是 对 自由 民主 人权 最 大 的 伤害                                                                         |
   | Target  | regardless of its aims or however highsounding its reasons are  war can always do the biggest harm to freedom  democracy  and human rights   |
   | Predict | in by by in by by by by a on military of at we of and military power military of of                                                          |
   +---------+----------------------------------------------------------------------------------------------------------------------------------------------+
   ```

   Again, the results are **just a reference**.

   

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