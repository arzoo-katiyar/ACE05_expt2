#export OUTPUT_DIR=ace05-onlyentity-top20
export OUTPUT_DIR=ace05-onlyentity-top20
export BATCH_SIZE=1
#32
export NUM_EPOCHS=20
export SAVE_STEPS=1076
export SEED=1
export BERT_MODEL=spanbert_hf_base/pytorch_model.bin
#export BERT_MODEL=ace05-onlyentity
export CONFIG_PATH=spanbert_hf_base/config.json
#export CONFIG_PATH=ace05-onlyentity/config.json
export TOKENIZER_PATH=bert-base-cased
#export TOKENIZER_PATH=ace05-onlyentity
export MAX_TRAIN_LENGTH=500
export MAX_TEST_LENGTH=500
#128

CUDA_VISIBLE_DEVICES=0 python3 run_onlyentity.py --data_dir ./data/ \
		    --model_type bert \
		    --labels ./data/labels.txt \
		    --model_name_or_path $BERT_MODEL \
		    --config_name $CONFIG_PATH \
		    --tokenizer_name $TOKENIZER_PATH \
		    --output_dir $OUTPUT_DIR \
		    --max_train_seq_length  $MAX_TRAIN_LENGTH \
		    --max_test_seq_length $MAX_TEST_LENGTH \
		    --num_train_epochs $NUM_EPOCHS \
		    --per_gpu_train_batch_size $BATCH_SIZE \
		    --save_steps $SAVE_STEPS \
		    --seed $SEED \
		    --do_train \
		    --do_eval \
		    --do_predict \
		    --overwrite_cache \
		    --overwrite_output_dir \
		    --fp16 \
		    --evaluate_during_training \
		    --logging_steps 1076
