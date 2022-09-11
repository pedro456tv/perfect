for dataset in sst-5; do
    for sample_size in 8 64; do
        for data_seed in 0 1; do
        cat > configs/setfit_template.json <<EOF
{
"task": "$dataset", 
"max_seq_length": 256, 
"per_device_train_batch_size": 32,
"per_device_eval_batch_size": 512,
"K": $sample_size,
"data_dir": "datasets_processed",
"output_dir": "outputs",
"model_name_or_path": "roberta-large",
"do_train": true,
"do_eval": true,
"do_predict": true,
"learning_rate": 1e-4,
"metric_name": "accuracy",
"max_steps": 6, 
"eval_steps": 6, 
"save_steps": 6,
"data_seed": $data_seed,
"seed": 1,
"load_best_model_at_end": true,
"metric_for_best_model": "average",
"greater_is_better": true,
"evaluation_strategy": "steps",
"save_strategy": "steps",
"save_total_limit": 1,
"overwrite_output_dir": true,
"soft_pet": true,
"soft_pet_loss": "extra_tokens",
"extra_without_original": true,
"extra_tokens_init": "random", 
"soft_mask_labels_learning_rate": 1e-1,
"adapter_tune": true,
"tune_layernorms": true,
"add_layer_norm_after_adapter": false, 
"add_layer_norm_before_adapter": false,
"train_in_batch": true,
"add_adapter_after_attention": false, 
"add_adapter_after_feedforward": true,
"extra_embd_initializer_range": 1e-4,
"overwrite_cache": true,
"prototypical_eval": true,
"eval_soft_pet_aggregation": "max",
"prototypical_similarity": "euc",
"token_hinge_loss": true,
"mask_position": "1",
"fp16": true
}
EOF
        echo "Running script with dataset $dataset and sample size $sample_size and seed $data_seed"
        python run_clm.py configs/setfit_template.json
        done
    done
done
