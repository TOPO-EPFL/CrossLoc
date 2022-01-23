# CrossLoc learning full steps

This notes contain the steps for full training, validation and evaluation steps for the proposed CrossLoc algorithm. We keep all the original command lines used for reproducing our results. Please refer to the [main README](README.md) for other steps.

### **Encoders Pretraining**

* Training: task-agnostic `LHS-sim` synthetic data is used (at `train_sim` folder).

```bash
# specify checkpoint weight output path
export CKPT_DIR=$(pwd)/ckpt-weights

# pretrain encoders with LHS-sim data for urbanscape
bash script_clean_training/encoder_pretrain.sh urbanscape coord FULL 1.0 in_place 0.0 mle 0
bash script_clean_training/encoder_pretrain.sh urbanscape depth FULL 1.0 in_place 0.0 mle 0
bash script_clean_training/encoder_pretrain.sh urbanscape normal FULL 1.0 in_place 0.0 mle 0
bash script_clean_training/encoder_pretrain.sh urbanscape semantics FULL 1.0 in_place 0.0 none 0

# pretrain encoders with LHS-sim data for naturescape
bash script_clean_training/encoder_pretrain.sh naturescape coord FULL 1.0 in_place 0.0 mle 0
bash script_clean_training/encoder_pretrain.sh naturescape depth FULL 1.0 in_place 0.0 mle 0
bash script_clean_training/encoder_pretrain.sh naturescape normal FULL 1.0 in_place 0.0 mle 0
bash script_clean_training/encoder_pretrain.sh naturescape semantics FULL 1.0 in_place 0.0 none 0

# pretrain encoders with in-place sim-to-real pairs for urbanscape [no LHS-sim pretraining, for ablation study]
bash script_clean_training/encoder_pretrain_pairwise_only.sh urbanscape coord FULL 0.0 in_place 1.0 mle 0
bash script_clean_training/encoder_pretrain_pairwise_only.sh urbanscape depth FULL 0.0 in_place 1.0 mle 0
bash script_clean_training/encoder_pretrain_pairwise_only.sh urbanscape normal FULL 0.0 in_place 1.0 mle 0
bash script_clean_training/encoder_pretrain_pairwise_only.sh urbanscape semantics FULL 0.0 in_place 1.0 none 0

# pretrain encoders with in-place real only data for urbanscape [no LHS-sim pretraining, for ablation study]
bash script_clean_training/encoder_pretrain_real_only.sh urbanscape coord FULL 0.0 in_place 1.0 mle 0
bash script_clean_training/encoder_pretrain_real_only.sh urbanscape depth FULL 0.0 in_place 1.0 mle 0
bash script_clean_training/encoder_pretrain_real_only.sh urbanscape normal FULL 0.0 in_place 1.0 mle 0
bash script_clean_training/encoder_pretrain_real_only.sh urbanscape semantics FULL 0.0 in_place 1.0 none 0
```

* Checkpoint selection: we evaluate the model performance on the validation set (at `val_sim` folder) and select the checkpoint models for later training tasks.

```bash
# specify checkpoint weight output path
export CKPT_DIR=$(pwd)/ckpt-weights/$TASK_DIR
# please specify $TASK_DIR for each task, e.g., naturescape-coord-sclean_training-unc-MLE-e100-lr0.0002-sim_only-sc1.00
# otherwise, the validation script may not load the network weight properly

# select model weight based on validation set performance for urbanscape
bash script_clean_validation/validate_encoder_pretrain.sh urbanscape coord FULL mle 0
bash script_clean_validation/validate_encoder_pretrain.sh urbanscape depth FULL mle 0
bash script_clean_validation/validate_encoder_pretrain.sh urbanscape normal FULL mle 0
bash script_clean_validation/validate_encoder_pretrain.sh urbanscape semantics FULL none 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details

# select model weight based on validation set performance for naturescape
bash script_clean_validation/validate_encoder_pretrain.sh naturescape coord FULL mle 0
bash script_clean_validation/validate_encoder_pretrain.sh naturescape depth FULL mle 0
bash script_clean_validation/validate_encoder_pretrain.sh naturescape normal FULL mle 0
bash script_clean_validation/validate_encoder_pretrain.sh naturescape semantics FULL none 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details

# select model weight based on validation set performance for in-place sim-to-real pairs for urbanscape [no LHS-sim pretraining, for ablation study]
bash script_clean_validation/validate_encoder_pretrain_pairwise_only.sh urbanscape coord FULL mle 0
bash script_clean_validation/validate_encoder_pretrain_pairwise_only.sh urbanscape depth FULL mle 0
bash script_clean_validation/validate_encoder_pretrain_pairwise_only.sh urbanscape normal FULL mle 0
bash script_clean_validation/validate_encoder_pretrain_pairwise_only.sh urbanscape semantics FULL none 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details

# select model weight based on validation set performance for in-place real only data for urbanscape [no LHS-sim pretraining, for ablation study]
bash script_clean_validation/validate_encoder_pretrain_real_only.sh urbanscape coord FULL mle 0
bash script_clean_validation/validate_encoder_pretrain_real_only.sh urbanscape depth FULL mle 0
bash script_clean_validation/validate_encoder_pretrain_real_only.sh urbanscape normal FULL mle 0
bash script_clean_validation/validate_encoder_pretrain_real_only.sh urbanscape semantics FULL none 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details
```

## Encoders Fine-tuning

* Training: to fine-tune the encoders with real-synthetic paired date. Note that the pretrained encoders' weights must be spcified in the script. Check the variable `ENC_PRETRAINED` in the `encoder_finetune.sh` script for detailed setup.

```bash
# specify checkpoint weight output path
export CKPT_DIR=$(pwd)/ckpt-weights

# finetune encoders with in-place sim-to-real pairs for urbanscape [using 100% LHS-pretrained weights]
bash script_clean_training/encoder_finetune.sh urbanscape coord FULL 1.0 in_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape depth FULL 1.0 in_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape normal FULL 1.0 in_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape semantics FULL 1.0 in_place 1.0 none 0

# finetune encoders with out-of-place sim-to-real pairs for urbanscape [using 100% LHS-pretrained weights]
bash script_clean_training/encoder_finetune.sh urbanscape coord FULL 1.0 out_of_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape depth FULL 1.0 out_of_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape normal FULL 1.0 out_of_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape semantics FULL 1.0 out_of_place 1.0 none 0

# finetune encoders with in-place sim-to-real pairs for naturescape [using 100% LHS-pretrained weights]
bash script_clean_training/encoder_finetune.sh naturescape coord FULL 1.0 in_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh naturescape depth FULL 1.0 in_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh naturescape normal FULL 1.0 in_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh naturescape semantics FULL 1.0 in_place 1.0 none 0

# finetune encoders with out-of-place sim-to-real pairs for naturescape [using 100% LHS-pretrained weights]
bash script_clean_training/encoder_finetune.sh naturescape coord FULL 1.0 out_of_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh naturescape depth FULL 1.0 out_of_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh naturescape normal FULL 1.0 out_of_place 1.0 mle 0
bash script_clean_training/encoder_finetune.sh naturescape semantics FULL 1.0 out_of_place 1.0 none 0

# finetune encoders with fractional in-place sim-to-real pairs for urbanscape [using 100% LHS-pretrained weights, ablation study]
# 25% sim-to-real pairs
bash script_clean_training/encoder_finetune.sh urbanscape coord FULL 1.0 in_place 0.25 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape depth FULL 1.0 in_place 0.25 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape normal FULL 1.0 in_place 0.25 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape semantics FULL 1.0 in_place 0.25 none 0

# 50% sim-to-real pairs
bash script_clean_training/encoder_finetune.sh urbanscape coord FULL 1.0 in_place 0.50 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape depth FULL 1.0 in_place 0.50 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape normal FULL 1.0 in_place 0.50 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape semantics FULL 1.0 in_place 0.50 none 0

# 75% sim-to-real pairs
bash script_clean_training/encoder_finetune.sh urbanscape coord FULL 1.0 in_place 0.75 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape depth FULL 1.0 in_place 0.75 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape normal FULL 1.0 in_place 0.75 mle 0
bash script_clean_training/encoder_finetune.sh urbanscape semantics FULL 1.0 in_place 0.75 none 0
```

* Checkpoint selection: again, we evaluate the model performance on the validation set (now at `val_drone_real` folder) and select the checkpoint models for later training tasks.

```bash
# specify checkpoint weight output path
export CKPT_DIR=$(pwd)/ckpt-weights/$TASK_DIR
# please specify $TASK_DIR for each task, e.g., naturescape-coord-sclean_training_pt1.00-unc-MLE-e800-lr0.0001-pairs-ip-rc1.00-finetune
# otherwise, the validation script may not load the network weight properly

# select model weight based on validation set performance for urbanscape data
# please change the $TASK_DIR and repeat for in-place and out-of-place scenes
export MIN_CKPT_ITER=800000  # use export MIN_CKPT_ITER=800000 for the 3x ablation studies
bash script_clean_validation/validate_encoder_finetune.sh urbanscape coord FULL mle 0
bash script_clean_validation/validate_encoder_finetune.sh urbanscape depth FULL mle 0
bash script_clean_validation/validate_encoder_finetune.sh urbanscape normal FULL mle 0
bash script_clean_validation/validate_encoder_finetune.sh urbanscape semantics FULL none 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details

# select model weight based on validation set performance naturescape data
# please change the $TASK_DIR and repeat for in-place and out-of-place scenes
export MIN_CKPT_ITER=1000000
bash script_clean_validation/validate_encoder_finetune.sh naturescape coord FULL mle 0
bash script_clean_validation/validate_encoder_finetune.sh naturescape depth FULL mle 0
bash script_clean_validation/validate_encoder_finetune.sh naturescape normal FULL mle 0
bash script_clean_validation/validate_encoder_finetune.sh naturescape semantics FULL none 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details
```

### Decoders Fine-tuning

* Training: to reuse the multiple fine-tuned encoders and fine-tune the decoder with real-synthetic paired date. Note that the encoders' weights from the last step must be spcified in the script. Check the variable `ENC_COORD`, `ENC_DEPTH`, `ENC_NORMAL`  and `ENC_SEMANTICS` in the `decoder_finetune.sh` script for detailed setup.

```bash
# specify checkpoint weight output path
export CKPT_DIR=$(pwd)/ckpt-weights

# finetune decoder with in-place sim-to-real pairs for urbanscape 
# [using 100% LHS-pretrained + sim-to-real paired data fine-tuned encoders]
bash script_clean_training/decoder_finetune.sh urbanscape coord FULL 1.0 in_place 1.0 0.0 in_place 1.0 mle 0
bash script_clean_training/decoder_finetune_plus_semantics.sh urbanscape coord FULL 1.0 in_place 1.0 0.0 in_place 1.0 mle 0

# finetune decoder with out-of-place sim-to-real pairs for urbanscape 
# [using 100% LHS-pretrained + sim-to-real paired data fine-tuned encoders]
bash script_clean_training/decoder_finetune.sh urbanscape coord FULL 1.0 out_of_place 1.0 0.0 out_of_place 1.0 mle 0
bash script_clean_training/decoder_finetune_plus_semantics.sh urbanscape coord FULL 1.0 out_of_place 1.0 0.0 out_of_place 1.0 mle 0

# finetune decoder with in-place sim-to-real pairs for naturescape 
# [using 100% LHS-pretrained + sim-to-real paired data fine-tuned encoders]
bash script_clean_training/decoder_finetune.sh naturescape coord FULL 1.0 in_place 1.0 0.0 in_place 1.0 mle 0
bash script_clean_training/decoder_finetune_plus_semantics.sh naturescape coord FULL 1.0 in_place 1.0 0.0 in_place 1.0 mle 0

# finetune decoder with out-of-place sim-to-real pairs for naturescape 
# [using 100% LHS-pretrained + sim-to-real paired data fine-tuned encoders]
bash script_clean_training/decoder_finetune.sh naturescape coord FULL 1.0 out_of_place 1.0 0.0 out_of_place 1.0 mle 0
bash script_clean_training/decoder_finetune_plus_semantics.sh naturescape coord FULL 1.0 out_of_place 1.0 0.0 out_of_place 1.0 mle 0

# finetune decoder with fractional in-place sim-to-real pairs for urbanscape 
# [using 100% LHS-pretrained weights + partial sim-to-real paired data fine-tuned encoders, ablation study]
# 25% sim-to-real pairs
bash script_clean_training/decoder_finetune.sh urbanscape coord FULL 1.0 in_place 0.25 0.0 in_place 0.25 mle 0
bash script_clean_training/decoder_finetune_plus_semantics.sh urbanscape coord FULL 1.0 in_place 0.25 0.0 in_place 0.25 mle 0

# 50% sim-to-real pairs
bash script_clean_training/decoder_finetune.sh urbanscape coord FULL 1.0 in_place 0.50 0.0 in_place 0.50 mle 0
bash script_clean_training/decoder_finetune_plus_semantics.sh urbanscape coord FULL 1.0 in_place 0.50 0.0 in_place 0.50 mle 0

# 75% sim-to-real pairs
bash script_clean_training/decoder_finetune.sh urbanscape coord FULL 1.0 in_place 0.75 0.0 in_place 0.75 mle 0
bash script_clean_training/decoder_finetune_plus_semantics.sh urbanscape coord FULL 1.0 in_place 0.75 0.0 in_place 0.75 mle 0

# using no LHS-sim data, but pairwise-only or real-only data
bash script_clean_training/decoder_finetune_pairwise_only.sh urbanscape coord FULL 0.0 in_place 1.0 0.0 in_place 1.0 mle 0
bash script_clean_training/decoder_finetune_real_only.sh urbanscape coord FULL 0.0 in_place 1.0 0.0 in_place 1.0 mle 0
```

* Checkpoint selection: again, we evaluate the model performance on the validation set (now at `val_drone_real` folder) and select the checkpoint models for later training tasks.

```bash
# specify checkpoint weight output path
export CKPT_DIR=$(pwd)/ckpt-weights/$TASK_DIR
# please specify $TASK_DIR for each task, e.g., naturescape-coord-decoder_coord_free_depth_normal-senc-pt1.00-ip-ft1.00-unc-MLE-e1000-lr0.0001-pairwise-ip-rc1.00
# otherwise, the validation script may not load the network weight properly

# select model weight based on validation set performance for urbanscape data
# please change the $TASK_DIR and repeat for in-place and out-of-place scenes
export MIN_CKPT_ITER=1000000  # in-place, use export MIN_CKPT_ITER=300000 for the partial-finetuning ablation studies
export MIN_CKPT_ITER=500000   # out-of-place
bash script_clean_validation/validate_decoder_finetune.sh urbanscape coord FULL mle 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details

# select model weight based on validation set performance naturescape data
# please change the $TASK_DIR and repeat for in-place and out-of-place scenes
export MIN_CKPT_ITER=1000000  # in-place
export MIN_CKPT_ITER=200000   # out-of-place
bash script_clean_validation/validate_decoder_finetune.sh naturescape coord FULL mle 0
# select the checkpoint from the generated path, see script_clean_validation/select_ckpt.py for details
```

### Performance Testing

Using the model weight selected above, we now run the testing script to evaluate the model's final performance on **testing set**.

```bash
# specify the specific weight path, change this accordingly before running each line
export WEIGHT_PATH=YOUR_PATH

# urbanscape, in-place scene
python3 test_single_task.py urbanscape --task coord --uncertainty mle --section test_drone_real --network_in ${WEIGHT_PATH}
# urbanscape, out-of-place scene
python3 test_single_task.py urbanscape --task coord --uncertainty mle --section test_oop_drone_real --network_in ${WEIGHT_PATH}

# naturescape, in-place scene
python3 test_single_task.py naturescape --task coord --uncertainty mle --section test_drone_real --network_in ${WEIGHT_PATH}
# naturescape, out-of-place scene
python3 test_single_task.py naturescape --task coord --uncertainty mle --section test_oop_drone_real --network_in ${WEIGHT_PATH}
```
