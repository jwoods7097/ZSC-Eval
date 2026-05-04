for agent in cole mep hsp fcp
do
    for env in random0 random3
    do
        traj_dir="$HOME/ZSC/results/Overcooked/${env}/population/selfplay-${env}-${agent}-seed1/trajs/${env}"

        # convert trajectories to pickle
        python eval/detect_int_llm.py \
            --traj_dir "$traj_dir" \
            --out_csv "eval/selfplay_results/${env}_${agent}_seed1_llm.csv" \
            --out_jsonl "eval/selfplay_results/${env}_${agent}_seed1_llm.jsonl"
    done
done