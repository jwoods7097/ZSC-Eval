export POLICY_POOL=../../policy_pool

python eval/eval.py \
  --env_name Overcooked \
  --algorithm_name population \
  --experiment_name selfplay-random3-cole-s50-seed1 \
  --layout_name random3 \
  --num_agents 2 \
  --seed 1 \
  --episode_length 400 \
  --n_eval_rollout_threads 100 \
  --eval_episodes 100 \
  --eval_stochastic \
  --dummy_batch_size 2 \
  --use_proper_time_limits \
  --population_yaml_path eval/selfplay_ymls/random3_cole_s50_seed1.yml \
  --population_size 2 \
  --overcooked_version old \
  --eval_result_path eval/results/random3/cole/selfplay-random3-cole-s50-seed1.json \
  --agent0_policy_name cole_self_a \
  --agent1_policy_name cole_self_b \
  --store_traj