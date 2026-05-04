export POLICY_POOL=../../policy_pool

for agent in cole mep hsp fcp
do
  if [ "$agent" = "cole" ]
  then
    pop=50
  else
    pop=36
  fi

  for env in random0 random3
  do
    cat << EOF > eval/selfplay_ymls/config.yml
self_a:
  policy_config_path: ${env}/policy_config/rnn_policy_config.pkl
  featurize_type: ppo
  train: False
  model_path:
    actor: ${env}/${agent}/s2/${agent}-S2-s${pop}/1.pt

self_b:
  policy_config_path: ${env}/policy_config/rnn_policy_config.pkl
  featurize_type: ppo
  train: False
  model_path:
    actor: ${env}/${agent}/s2/${agent}-S2-s${pop}/1.pt
EOF

    python eval/eval.py \
      --env_name Overcooked \
      --algorithm_name population \
      --experiment_name "selfplay-${env}-${agent}-seed1" \
      --layout_name $env \
      --num_agents 2 \
      --seed 1 \
      --episode_length 200 \
      --n_eval_rollout_threads 100 \
      --eval_episodes 100 \
      --eval_stochastic \
      --dummy_batch_size 2 \
      --use_proper_time_limits \
      --population_yaml_path "eval/selfplay_ymls/config.yml" \
      --population_size 2 \
      --overcooked_version old \
      --eval_result_path "eval/results/${env}/${agent}/selfplay-${env}-${agent}-seed1.json" \
      --agent0_policy_name self_a \
      --agent1_policy_name self_b \
      --store_traj
  done
done