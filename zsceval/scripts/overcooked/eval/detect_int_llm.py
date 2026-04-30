import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")
client = OpenAI()


SYSTEM_PROMPT = """
You evaluate interdependencies in cooperative multi-agent trajectories.

Definitions:
- An interdependence occurs when one agent's earlier action creates a condition, resource, object state, or opportunity that enables the other agent's later action.
- constructive: a real interdependence that contributes to task progress or a later reward/goal and is not merely looping or redundant.
- non_constructive: a real interdependence that is redundant, wasted, misaligned, or does not contribute to task progress.
- looping: an object/resource is passed around or returned without useful progress.
- irrelevant: dependency-like interaction unrelated to the task goal.
- no_dependency: do not count it.

You will receive compressed state-change lines from one episode.
Estimate episode-level counts:
  cons_int, non_cons_int, loop_int, irr_int

Be conservative. Count only clear cross-agent dependencies.
Return JSON only.
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "task_rew": {"type": "number"},
        "cons_int": {"type": "integer"},
        "non_cons_int": {"type": "integer"},
        "loop_int": {"type": "integer"},
        "irr_int": {"type": "integer"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": [
        "task_rew",
        "cons_int",
        "non_cons_int",
        "loop_int",
        "irr_int",
        "confidence",
        "notes",
    ],
    "additionalProperties": False,
}


def compact_object(obj: Any) -> Any:
    """Make objects shorter and JSON-stable."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        out = {}
        for k in ["name", "position", "state"]:
            if k in obj and obj[k] is not None:
                out[k] = obj[k]
        return out

    return obj


def summarize_state(state: dict) -> dict:
    """
    Keep only the parts usually relevant for cooperation:
    players' held objects and object locations/states.

    This is intentionally lightweight. It does not use PDDL.
    """
    players = []
    for i, p in enumerate(state.get("players", [])):
        players.append({
            "id": i,
            "pos": p.get("position"),
            "ori": p.get("orientation"),
            "held": compact_object(p.get("held_object")),
        })

    objects = state.get("objects", {})
    compact_objects = {}

    if isinstance(objects, dict):
        iterable = objects.items()
    else:
        # Some ZSC-Eval versions store objects as a list.
        iterable = []
        for obj in objects:
            pos = obj.get("position", "unknown")
            iterable.append((str(pos), obj))

    for key, obj in iterable:
        compact_objects[str(key)] = compact_object(obj)

    return {
        "players": players,
        "objects": compact_objects,
        "order_list": state.get("order_list"),
    }


def json_dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def describe_diff(t: int, s0: dict, action: Any, reward: float, s1: dict) -> list[str]:
    """
    Produce short human-readable lines from state_t -> state_t+1.

    We mainly track:
    - held object changes
    - objects appearing/disappearing/changing
    - reward events
    """
    before = summarize_state(s0)
    after = summarize_state(s1)

    lines = []

    # Player held-object changes.
    for p0, p1 in zip(before["players"], after["players"]):
        if json_dumps(p0.get("held")) != json_dumps(p1.get("held")):
            lines.append(
                f"t={t}: agent {p0['id']} held changed "
                f"from {p0.get('held')} to {p1.get('held')}; "
                f"joint_action={action}; reward={reward}"
            )

    # Object map changes.
    obj0 = before["objects"]
    obj1 = after["objects"]

    keys0 = set(obj0)
    keys1 = set(obj1)

    for k in sorted(keys1 - keys0):
        lines.append(
            f"t={t}: object appeared at {k}: {obj1[k]}; "
            f"joint_action={action}; reward={reward}"
        )

    for k in sorted(keys0 - keys1):
        lines.append(
            f"t={t}: object disappeared from {k}: {obj0[k]}; "
            f"joint_action={action}; reward={reward}"
        )

    for k in sorted(keys0 & keys1):
        if json_dumps(obj0[k]) != json_dumps(obj1[k]):
            lines.append(
                f"t={t}: object at {k} changed from {obj0[k]} to {obj1[k]}; "
                f"joint_action={action}; reward={reward}"
            )

    # Reward event.
    if reward:
        lines.append(
            f"t={t}: positive reward {reward}; joint_action={action}; "
            f"state_after_summary={after}"
        )

    return lines


def compress_trajectory(path: Path) -> tuple[float, list[str]]:
    with path.open("r") as f:
        traj = json.load(f)

    states = traj["ep_states"][0]
    actions = traj["ep_actions"][0]
    rewards = traj["ep_rewards"][0]

    task_rew = float(sum(rewards))
    lines = []

    n = min(len(actions), len(rewards), len(states) - 1)

    for t in range(n):
        lines.extend(
            describe_diff(
                t=t,
                s0=states[t],
                action=actions[t],
                reward=rewards[t],
                s1=states[t + 1],
            )
        )

    return task_rew, lines


def classify_episode(path: Path) -> dict:
    task_rew, lines = compress_trajectory(path)
    for line in lines:
        print(line)

    payload = {
        "trajectory_file": str(path),
        "task_goal_hint": (
            "Two agents cooperate. Positive reward indicates task progress/completion. "
            "In Overcooked-like trajectories, rewards usually come from delivered soups."
        ),
        "computed_task_rew": task_rew,
        "compressed_state_changes": lines,
    }

    response = client.responses.create(
        model=MODEL,
        temperature=0,
        input=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "InterdependenceEpisode",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    )

    result = json.loads(response.output_text)

    # Trust local reward accounting over model reward accounting.
    result["task_rew"] = task_rew
    result["trajectory_file"] = str(path)
    result["num_lines_sent"] = len(lines)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", required=True)
    parser.add_argument("--out_csv", default="obj_results_llm.csv")
    parser.add_argument("--out_jsonl", default="llm_episode_results.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    traj_paths = sorted(Path(args.traj_dir).glob("*.json"))
    if args.limit is not None:
        traj_paths = traj_paths[: args.limit]

    rows = []

    with open(args.out_jsonl, "w") as jf:
        for path in tqdm(traj_paths):
            result = classify_episode(path)
            jf.write(json.dumps(result) + "\n")

            rows.append([
                result["task_rew"],
                result["cons_int"],
                result["non_cons_int"],
                result["loop_int"],
                result["irr_int"],
            ])

    with open(args.out_csv, "w", newline="") as cf:
        writer = csv.writer(cf)
        # Match detect_int_proxy.py shape: no header.
        writer.writerows(rows)

    if rows:
        means = [
            sum(row[i] for row in rows) / len(rows)
            for i in range(5)
        ]
        print("Wrote:", args.out_csv)
        print("N:", len(rows))
        print("[task_rew, cons_int, non_cons_int, loop_int, irr_int] means:")
        print([round(x, 3) for x in means])


if __name__ == "__main__":
    main()