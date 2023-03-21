DATA_DIR=../data/swissdox/new_vocab/bin
MODEL_NAME=swissbert_new_vocab

fairseq-train \
  $DATA_DIR \
  --save-dir ../models/swissbert/$MODEL_NAME \
  --finetune-from-model ../models/xmod.base.81.1M/model.pt \
  --user-dir ../fairseq_additions \
  --arch swissbert_base \
  --task multilingual_masked_lm_xmod \
  --train-new-embeddings \
  --old-vocab-path ../vocab/xlm.sentencepiece.bpe.model \
  --new-vocab-path ../vocab/swissbert.v1.model \
  --multilang-sampling-alpha 0.3 \
  --init-languages "de_DE->de_CH,fr_XX->fr_CH,it_IT->it_CH" \
  --prune-languages \
  --sample-break-mode none \
  --criterion masked_lm \
  --seed 913 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --optimizer adam \
  --weight-decay 0.01 \
  --adam-betas "(0.9, 0.999)" \
  --adam-eps 1e-06 \
  --lr-scheduler polynomial_decay \
  --warmup-updates 36568 \
  --total-num-update 365680 \
  --tokens-per-sample 512 \
  --skip-invalid-size-inputs-valid-test \
  --clip-norm 1.0 \
  --lr 0.0007 \
  --update-freq 12 \
  --batch-size 12 \
  --log-interval 100 \
  --save-interval-updates 2000 \
  --max-epoch 10 \
  --ddp-backend no_c10d \
  --fp16 \
  --tensorboard-logdir ../logs/$MODEL_NAME
