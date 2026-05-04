#!/usr/bin/env python3
"""
Lightweight LLM interdependence evaluator for Biswas-style Overcooked trajectory JSONs.

Input trajectory format expected:
{
  "ep_states": [[state_0, state_1, ...]],
  "ep_actions": [[[a0, a1], ...]],
  "ep_rewards": [[0, 0, 20, ...]],
  "mdp_params": [{...}]                   # optional
}

This version is intentionally lightweight but Overcooked-aware:
1. Computes task reward locally as sum(ep_rewards[0]).
2. Converts only agent-level changes caused by INTERACT actions into natural language.
3. Does not send raw joint actions, raw reward values, movement actions, object-map diffs,
   or internal object-state changes to the LLM.
4. Tracks onion identities as onion_1, onion_2, ... and includes those IDs in event lines.
5. Builds an onion lineage summary and a soup composition summary, listing which 3 onions
   went into each soup whenever this can be inferred.
6. Prompts the LLM with failure-case checks inspired by Biswas detect_int_proxy.py:
   - only cross-agent dependencies from interact actions,
   - goal-reaching object/resource chains,
   - giver/receiver looping checks,
   - non-goal-reaching dependencies.
7. Writes a detect_int_proxy-compatible CSV with no header:
   task_rew, cons_int, non_cons_int, loop_int, irr_int

Example:
  export OPENAI_API_KEY="..."
  export OPENAI_MODEL="gpt-4.1-mini"

  python llm_interdeps_light_nl_interact_onion_ids.py \
    --traj_dir eval/trajectories/random0/cole/seed1 \
    --out_csv obj_results_llm.csv \
    --out_jsonl llm_episode_results.jsonl \
    --limit 10

Inspect natural-language extraction without calling the API:
  python llm_interdeps_light_nl_interact_onion_ids.py --traj_dir PATH --dry_run --limit 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")


SYSTEM_PROMPT = """
You evaluate interdependencies in Overcooked cooperative multi-agent trajectories.

The input contains only natural-language summaries of events caused by agents' INTERACT actions. Movement/stay actions, raw action IDs, numeric rewards, object-map diffs, and internal object-state changes have already been removed.

Important Overcooked context:
- Two agents cooperate to make and deliver soups.
- Onions are tracked with explicit IDs like onion_1, onion_2, etc.
- Soups are tracked with explicit IDs like soup_1, soup_2, etc.
- The input includes an onion lineage summary and a soup composition/delivery summary. Use those summaries to determine whether an onion, dish, or soup handoff was goal-reaching.
- A soup composition line such as "soup_1 used onions [onion_1, onion_2, onion_3]" means exactly those onions were used for that soup.
- A soup or onion is goal-reaching only if the corresponding soup is explicitly delivered within the trajectory. A filled, cooked, or completed-but-undelivered soup is NOT goal-reaching.

Definitions, following the spirit of Biswas et al.'s detector:
- Consider only cross-agent interdependencies: an earlier INTERACT by one agent creates, moves, or modifies an object/resource/opportunity, and a later INTERACT by the other agent uses that same object/resource/opportunity.
- The earlier agent is the giver. The later agent is the receiver.
- A receiver event should only count if it appears to require the giver-created resource. Example: agent 0 places onion_4 on a shared counter, then agent 1 later picks up onion_4 from that same shared counter.
- Count dish and soup handoffs separately from onion handoffs when they are explicit in the event log. They are separate object/resource dependencies, not double-counting.

Failure-case checks to apply before counting a constructive interdependence:
1. Looping check: count loop_int, not cons_int, if the same object/resource is passed back to the giver, cycles between agents, or the receiver appears to have already had that same object/resource before the supposed receiving action.
2. Goal-reaching check: count irr_int, not cons_int, if the object/resource involved in the dependency never contributes to an explicitly delivered soup. For onions, the specific onion ID must appear in a soup composition whose soup ID is delivered. For soup or dish handoffs, the specific soup/dish handoff must contribute to a delivered soup.
3. Non-constructive check: count non_cons_int for real cross-agent dependencies that are neither looping nor clearly non-goal-reaching, but are still wasted, misaligned, redundant, or do not clearly improve Overcooked task progress.
4. Constructive check: count cons_int only when the dependency is cross-agent, uses the same onion/soup/dish/resource ID or shared opportunity, is goal-reaching, and is non-looping.

Be conservative:
- Do not count independent same-agent progress.
- Do not count two agents merely doing useful things in parallel.
- Do not count a dependency if the later event could have happened independently without the earlier event.
- Do not infer dependencies from movement alone; movement has been omitted intentionally.
- Use onion IDs and soup IDs to avoid over-counting ambiguous handoffs.
- Do not count onions in undelivered soups as constructive, even if the soup was filled or cooked.
- Do not invent missing dish, soup, or onion handoffs.

Return JSON only with:
{
  "cons_int": integer,
  "non_cons_int": integer,
  "loop_int": integer,
  "irr_int": integer,
  "confidence": number,
  "notes": string
}

In notes, briefly explain the accounting, e.g. delivered-soup onion handoffs, delivered-soup dish/soup handoffs, undelivered-soup irrelevant handoffs, loops, and non-constructive cases. The numeric counts must match the notes.
""".strip()


SCHEMA = {
    "type": "object",
    "properties": {
        "cons_int": {"type": "integer", "minimum": 0},
        "non_cons_int": {"type": "integer", "minimum": 0},
        "loop_int": {"type": "integer", "minimum": 0},
        "irr_int": {"type": "integer", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "required": [
        "cons_int",
        "non_cons_int",
        "loop_int",
        "irr_int",
        "confidence",
        "notes",
    ],
    "additionalProperties": False,
}


# -----------------------------
# Action helpers
# -----------------------------


INTERACT_ACTION_INDEX = 5


def is_interact_action(action: Any) -> bool:
    """Return True only for Overcooked interact actions.

    Supports common trajectory encodings:
      - integer index 5
      - string "interact"
      - dict with action/action_name/action_type fields
      - raw old-backend action string in a joint action list
    """
    if action == INTERACT_ACTION_INDEX:
        return True

    if isinstance(action, str):
        return action.lower() == "interact"

    if isinstance(action, dict):
        raw = action.get("action") or action.get("action_name") or action.get("name")
        if isinstance(raw, str) and raw.lower() == "interact":
            return True
        # Some symbolic logs encode only action_type for interact-derived subtasks.
        if action.get("action_type") is not None and raw is None:
            return True

    return False


def action_for_agent(actions_t: Any, agent_id: int) -> Any:
    """Get one agent's action from a joint-action entry."""
    if actions_t is None:
        return None

    if isinstance(actions_t, (list, tuple)):
        if agent_id < len(actions_t):
            return actions_t[agent_id]
        return None

    if isinstance(actions_t, dict):
        for key in (agent_id, str(agent_id), f"agent_{agent_id}", f"Agent_{agent_id}"):
            if key in actions_t:
                return actions_t[key]
        return None

    return None


# -----------------------------
# JSON / object helpers
# -----------------------------


def json_dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def raw_obj_name(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    return str(name) if name is not None else None


def compact_object(obj: Any) -> Any:
    """Keep only object name and position. Drop internal object state.

    This intentionally removes fields such as soup ingredients/cook state, timers,
    and other internal states before anything is sent to the LLM.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        if obj.get("name") is not None:
            out["name"] = obj.get("name")
        if obj.get("position") is not None:
            out["position"] = obj.get("position")
        return out if out else None

    return obj


def obj_name(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    return str(name) if name is not None else None


def obj_brief(obj: Any) -> str:
    """Natural object description with no internal state."""
    if obj is None:
        return "nothing"

    if not isinstance(obj, dict):
        return repr(obj)

    name = obj.get("name", "object")
    return f"'{name}'"


def same_object_type(a: Any, b: Any) -> bool:
    return obj_name(a) is not None and obj_name(a) == obj_name(b)


def normalize_objects(objects: Any) -> dict[str, Any]:
    """Convert object containers to a stable dict keyed by location-ish string.

    Object internal state is stripped by compact_object().
    """
    if isinstance(objects, dict):
        out: dict[str, Any] = {}
        for key, value in objects.items():
            compact = compact_object(value)
            if compact is not None:
                out[str(key)] = compact
        return out

    if isinstance(objects, list):
        out = {}
        for obj in objects:
            compact = compact_object(obj)
            if compact is None:
                continue
            pos = compact.get("position", "unknown") if isinstance(compact, dict) else "unknown"
            out[str(pos)] = compact
        return out

    return {}


def normalize_raw_objects(objects: Any) -> dict[str, Any]:
    """Return raw objects keyed by location-ish string for internal tracking only.

    These raw objects are never sent to the LLM. They are used only to infer which
    pot/container changed when an onion is added or soup is picked up.
    """
    if isinstance(objects, dict):
        return {str(k): v for k, v in objects.items()}

    if isinstance(objects, list):
        out: dict[str, Any] = {}
        for obj in objects:
            if isinstance(obj, dict):
                pos = obj.get("position", "unknown")
                out[str(pos)] = obj
        return out

    return {}


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Keep only player inventory and shared object names/locations."""
    players = []
    for i, player in enumerate(state.get("players", [])):
        players.append(
            {
                "id": i,
                "held": compact_object(player.get("held_object")),
            }
        )

    return {
        "players": players,
        "objects": normalize_objects(state.get("objects", {})),
    }


# -----------------------------
# Onion and soup identity tracking
# -----------------------------


class OnionSoupTracker:
    """Best-effort lineage tracker for Overcooked onions and soups.

    Raw trajectories usually do not contain stable object IDs for onions. This
    tracker assigns IDs based on interact-derived object flow:
      - picking an onion from a dispenser/source creates a new onion ID;
      - placing/picking from a shared counter transfers that ID;
      - adding an onion to a pot assigns it to a soup ID;
      - a soup ID is associated with the three onion IDs that were added to it.

    The tracker may be approximate when multiple pots are used concurrently and
    the raw state does not expose enough pot information. It avoids exposing any
    raw internal object state to the LLM.
    """

    def __init__(self) -> None:
        self.next_onion_num = 1
        self.next_soup_num = 1

        self.agent_object_ids: dict[int, str | None] = defaultdict(lambda: None)
        self.shared_object_ids: dict[str, str] = {}

        self.onion_history: dict[str, list[str]] = defaultdict(list)
        self.soup_onions: dict[str, list[str]] = defaultdict(list)
        self.soup_history: dict[str, list[str]] = defaultdict(list)
        self.soup_location: dict[str, str] = {}
        self.soup_by_location: dict[str, str] = {}
        self.agent_soup_ids: dict[int, str | None] = defaultdict(lambda: None)
        self.completed_soups: list[str] = []

    def new_onion(self, t: int, note: str) -> str:
        onion_id = f"onion_{self.next_onion_num}"
        self.next_onion_num += 1
        self.onion_history[onion_id].append(f"t={t}: created/tracked when {note}")
        return onion_id

    def new_soup(self, location: str | None = None) -> str:
        soup_id = f"soup_{self.next_soup_num}"
        self.next_soup_num += 1
        if location is not None:
            self.soup_location[soup_id] = location
            self.soup_by_location[location] = soup_id
        return soup_id

    def get_or_create_soup_for_location(self, location: str | None) -> str:
        key = location or "unknown_pot"
        existing = self.soup_by_location.get(key)
        if existing is not None and len(self.soup_onions[existing]) < 3:
            return existing
        return self.new_soup(key)

    def soup_for_filling(self, location: str | None) -> str:
        # Prefer a completed soup at the changed pot/container location.
        if location is not None:
            soup_id = self.soup_by_location.get(location)
            if soup_id is not None:
                return soup_id

        # Otherwise choose the earliest completed soup that has not been picked up/delivered.
        for soup_id in self.completed_soups:
            return soup_id

        # Fallback: create an unknown soup.
        return self.new_soup(location or "unknown_pot")

    def pickup_onion_from_source(self, t: int, agent_id: int) -> str:
        onion_id = self.new_onion(t, f"agent {agent_id} picked up an onion from a dispenser/source")
        self.agent_object_ids[agent_id] = onion_id
        self.onion_history[onion_id].append(f"t={t}: agent {agent_id} picked it up from a dispenser/source")
        return onion_id

    def pickup_onion_from_shared(self, t: int, agent_id: int, loc: str) -> str:
        onion_id = self.shared_object_ids.pop(loc, None)
        if onion_id is None or not onion_id.startswith("onion_"):
            onion_id = self.new_onion(t, f"agent {agent_id} picked up an onion from shared location {loc}")
        self.agent_object_ids[agent_id] = onion_id
        self.onion_history[onion_id].append(f"t={t}: agent {agent_id} picked it up from shared location {loc}")
        return onion_id

    def place_onion_on_shared(self, t: int, agent_id: int, loc: str) -> str:
        onion_id = self.agent_object_ids.get(agent_id)
        if onion_id is None or not onion_id.startswith("onion_"):
            onion_id = self.new_onion(t, f"agent {agent_id} placed an onion on shared location {loc}")
        self.shared_object_ids[loc] = onion_id
        self.agent_object_ids[agent_id] = None
        self.onion_history[onion_id].append(f"t={t}: agent {agent_id} placed it on shared location {loc}")
        return onion_id

    def add_onion_to_soup(self, t: int, agent_id: int, location: str | None) -> tuple[str, str]:
        onion_id = self.agent_object_ids.get(agent_id)
        if onion_id is None or not onion_id.startswith("onion_"):
            onion_id = self.new_onion(t, f"agent {agent_id} added an onion to a pot/container")

        soup_id = self.get_or_create_soup_for_location(location)
        self.soup_onions[soup_id].append(onion_id)
        if location is not None:
            self.soup_location[soup_id] = location
            self.soup_by_location[location] = soup_id

        self.agent_object_ids[agent_id] = None
        self.onion_history[onion_id].append(f"t={t}: agent {agent_id} added it to {soup_id}")
        self.soup_history[soup_id].append(f"t={t}: agent {agent_id} added {onion_id}")

        if len(self.soup_onions[soup_id]) == 3 and soup_id not in self.completed_soups:
            self.completed_soups.append(soup_id)
            onions = ", ".join(self.soup_onions[soup_id])
            self.soup_history[soup_id].append(f"t={t}: {soup_id} now has three onions [{onions}]")

        return onion_id, soup_id

    def fill_soup(self, t: int, agent_id: int, location: str | None) -> str:
        soup_id = self.soup_for_filling(location)
        self.agent_soup_ids[agent_id] = soup_id
        self.agent_object_ids[agent_id] = soup_id
        self.soup_history[soup_id].append(f"t={t}: agent {agent_id} filled a dish with {soup_id}")
        if soup_id in self.completed_soups:
            self.completed_soups.remove(soup_id)
        return soup_id

    def deliver_soup(self, t: int, agent_id: int) -> str:
        soup_id = self.agent_soup_ids.get(agent_id) or self.agent_object_ids.get(agent_id)
        if soup_id is None or not str(soup_id).startswith("soup_"):
            soup_id = self.soup_for_filling(None)
        self.agent_soup_ids[agent_id] = None
        self.agent_object_ids[agent_id] = None
        self.soup_history[str(soup_id)].append(f"t={t}: agent {agent_id} delivered {soup_id}")
        return str(soup_id)

    def pickup_soup_from_shared(self, t: int, agent_id: int, loc: str) -> str:
        soup_id = self.shared_object_ids.pop(loc, None)
        if soup_id is None or not soup_id.startswith("soup_"):
            soup_id = self.soup_for_filling(None)
        self.agent_soup_ids[agent_id] = soup_id
        self.agent_object_ids[agent_id] = soup_id
        self.soup_history[soup_id].append(f"t={t}: agent {agent_id} picked up {soup_id} from shared location {loc}")
        return soup_id

    def place_soup_on_shared(self, t: int, agent_id: int, loc: str) -> str:
        soup_id = self.agent_soup_ids.get(agent_id) or self.agent_object_ids.get(agent_id)
        if soup_id is None or not str(soup_id).startswith("soup_"):
            soup_id = self.soup_for_filling(None)
        self.shared_object_ids[loc] = str(soup_id)
        self.agent_soup_ids[agent_id] = None
        self.agent_object_ids[agent_id] = None
        self.soup_history[str(soup_id)].append(f"t={t}: agent {agent_id} placed {soup_id} on shared location {loc}")
        return str(soup_id)

    def onion_summary_lines(self) -> list[str]:
        lines = []
        for onion_id in sorted(self.onion_history, key=lambda x: int(x.split("_")[1])):
            history = "; ".join(self.onion_history[onion_id])
            lines.append(f"{onion_id}: {history}.")
        return lines

    def soup_summary_lines(self) -> list[str]:
        lines = []
        def soup_sort_key(soup_id: str) -> int:
            try:
                return int(soup_id.split("_")[1])
            except Exception:
                return 10**9

        for soup_id in sorted(self.soup_onions, key=soup_sort_key):
            onions = self.soup_onions[soup_id]
            onions_text = ", ".join(onions) if onions else "unknown onions"
            made_text = "made" if len(onions) >= 3 else "partially assembled"
            history = "; ".join(self.soup_history.get(soup_id, []))
            lines.append(f"{soup_id}: {made_text} from onions [{onions_text}]. {history}.")

        # Include soup IDs created by filling/delivery fallback even if onion composition is unknown.
        for soup_id in sorted(set(self.soup_history) - set(self.soup_onions), key=soup_sort_key):
            history = "; ".join(self.soup_history.get(soup_id, []))
            lines.append(f"{soup_id}: onion composition unknown. {history}.")

        return lines


# -----------------------------
# Natural-language event extraction
# -----------------------------


def find_removed_matching_object(
    before_objects: dict[str, Any],
    after_objects: dict[str, Any],
    held_after: Any,
) -> tuple[str, Any] | None:
    """Detect pickup from a shared location."""
    removed_keys = set(before_objects) - set(after_objects)

    for key in sorted(removed_keys):
        obj = before_objects[key]
        if same_object_type(obj, held_after):
            return key, obj

    return None


def find_appeared_matching_object(
    before_objects: dict[str, Any],
    after_objects: dict[str, Any],
    held_before: Any,
) -> tuple[str, Any] | None:
    """Detect placement to a shared location."""
    appeared_keys = set(after_objects) - set(before_objects)

    for key in sorted(appeared_keys):
        obj = after_objects[key]
        if same_object_type(obj, held_before):
            return key, obj

    return None


def find_changed_soup_location(raw_before: dict[str, Any], raw_after: dict[str, Any]) -> str | None:
    """Best-effort internal-only inference of the pot/container location that changed.

    This can use raw object state internally, but those state details are never
    included in the LLM prompt.
    """
    keys = sorted(set(raw_before) | set(raw_after))
    for key in keys:
        before = raw_before.get(key)
        after = raw_after.get(key)
        if json_dumps(before) == json_dumps(after):
            continue
        before_name = raw_obj_name(before)
        after_name = raw_obj_name(after)
        if before_name == "soup" or after_name == "soup":
            return key
    return None


def naturalize_interact_held_change(
    t: int,
    agent_id: int,
    held_before: Any,
    held_after: Any,
    before_objects: dict[str, Any],
    after_objects: dict[str, Any],
    delivered: bool,
    soup_change_location: str | None,
    tracker: OnionSoupTracker,
) -> list[str]:
    """Convert only an INTERACT-caused inventory change into natural language.

    No movement, no raw action IDs, no raw reward, no object internal states, and
    no object-map state-change lines are emitted.
    """
    lines: list[str] = []

    before_empty = held_before is None
    after_empty = held_after is None

    if json_dumps(held_before) == json_dumps(held_after):
        # Interaction may have changed a pot/container internally, but the user
        # requested removal of internal object state changes, so we do not emit it.
        return lines

    # Pick up object.
    if before_empty and not after_empty:
        after_name = obj_name(held_after)
        match = find_removed_matching_object(before_objects, after_objects, held_after)

        if after_name == "onion":
            if match is not None:
                loc, _ = match
                onion_id = tracker.pickup_onion_from_shared(t, agent_id, loc)
                lines.append(
                    f"t={t}: agent {agent_id} picked up {onion_id} "
                    f"from location {loc}."
                )
            else:
                onion_id = tracker.pickup_onion_from_source(t, agent_id)
                lines.append(
                    f"t={t}: agent {agent_id} picked up {onion_id} "
                    f"from an onion dispenser"
                )
            return lines

        if after_name == "soup":
            if match is not None:
                loc, _ = match
                soup_id = tracker.pickup_soup_from_shared(t, agent_id, loc)
                onions = tracker.soup_onions.get(soup_id, [])
                onions_text = ", ".join(onions) if onions else "unknown onions"
                lines.append(
                    f"t={t}: agent {agent_id} picked up {soup_id} "
                    f"from location {loc}; {soup_id} used onions [{onions_text}]."
                )
            else:
                soup_id = tracker.fill_soup(t, agent_id, soup_change_location)
                onions = tracker.soup_onions.get(soup_id, [])
                onions_text = ", ".join(onions) if onions else "unknown onions"
                lines.append(
                    f"t={t}: agent {agent_id} picked up {soup_id}; "
                    f"{soup_id} used onions [{onions_text}]."
                )
            return lines

        lines.append(
            f"t={t}: agent {agent_id} picked up {obj_brief(held_after)}."
        )
        return lines

    # Put down, use, or deliver object.
    if not before_empty and after_empty:
        before_name = obj_name(held_before)

        if delivered and before_name in {"soup", "dish"}:
            soup_id = tracker.deliver_soup(t, agent_id)
            onions = tracker.soup_onions.get(soup_id, [])
            onions_text = ", ".join(onions) if onions else "unknown onions"
            lines.append(
                f"t={t}: agent {agent_id} delivered {soup_id} and achieved the goal; "
                f"{soup_id} used onions [{onions_text}]."
            )
            return lines

        appeared = find_appeared_matching_object(before_objects, after_objects, held_before)
        if appeared is not None:
            loc, _ = appeared
            if before_name == "onion":
                onion_id = tracker.place_onion_on_shared(t, agent_id, loc)
                lines.append(
                    f"t={t}: agent {agent_id} placed {onion_id} "
                    f"at location {loc}."
                )
                return lines
            if before_name == "soup":
                soup_id = tracker.place_soup_on_shared(t, agent_id, loc)
                onions = tracker.soup_onions.get(soup_id, [])
                onions_text = ", ".join(onions) if onions else "unknown onions"
                lines.append(
                    f"t={t}: agent {agent_id} placed {soup_id} "
                    f"at location {loc}; {soup_id} used onions [{onions_text}]."
                )
                return lines

            lines.append(
                f"t={t}: agent {agent_id} placed {obj_brief(held_before)} "
                f"at location {loc}."
            )
            return lines

        # Do not describe internal target object state; keep only the agent-level
        # interact event and object name/ID.
        if before_name == "onion":
            onion_id, soup_id = tracker.add_onion_to_soup(t, agent_id, soup_change_location)
            onions = tracker.soup_onions.get(soup_id, [])
            onions_text = ", ".join(onions)
            if len(onions) == 3:
                lines.append(
                    f"t={t}: agent {agent_id} added {onion_id} to {soup_id}; "
                    f"{soup_id} now uses onions [{onions_text}]."
                )
            else:
                lines.append(
                    f"t={t}: agent {agent_id} added {onion_id} to {soup_id}; "
                    f"{soup_id} currently has onions [{onions_text}]."
                )
        elif before_name == "dish":
            soup_id = tracker.fill_soup(t, agent_id, soup_change_location)
            onions = tracker.soup_onions.get(soup_id, [])
            onions_text = ", ".join(onions) if onions else "unknown onions"
            lines.append(
                f"t={t}: agent {agent_id} used a dish to collect {soup_id}; "
                f"{soup_id} used onions [{onions_text}]."
            )
        else:
            lines.append(
                f"t={t}: agent {agent_id} used {obj_brief(held_before)}."
            )
        return lines

    # Held object changed directly, e.g. dish -> soup.
    if not before_empty and not after_empty:
        before_name = obj_name(held_before)
        after_name = obj_name(held_after)

        if before_name == "dish" and after_name == "soup":
            soup_id = tracker.fill_soup(t, agent_id, soup_change_location)
            onions = tracker.soup_onions.get(soup_id, [])
            onions_text = ", ".join(onions) if onions else "unknown onions"
            lines.append(
                f"t={t}: agent {agent_id} filled a dish with {soup_id}; "
                f"{soup_id} used onions [{onions_text}]."
            )
        else:
            lines.append(
                f"t={t}: agent {agent_id} changed held object from "
                f"{obj_brief(held_before)} to {obj_brief(held_after)}."
            )
        return lines

    return lines


def describe_interact_diff(
    t: int,
    s0: dict[str, Any],
    actions_t: Any,
    reward: float,
    s1: dict[str, Any],
    tracker: OnionSoupTracker,
) -> list[str]:
    """Produce natural-language lines only for agents whose action was interact."""
    before = summarize_state(s0)
    after = summarize_state(s1)

    before_objects = before["objects"]
    after_objects = after["objects"]
    raw_before_objects = normalize_raw_objects(s0.get("objects", {}))
    raw_after_objects = normalize_raw_objects(s1.get("objects", {}))
    soup_change_location = find_changed_soup_location(raw_before_objects, raw_after_objects)
    delivered = reward > 0

    lines: list[str] = []

    for p0, p1 in zip(before["players"], after["players"]):
        agent_id = int(p0["id"])
        action = action_for_agent(actions_t, agent_id)
        if not is_interact_action(action):
            continue

        lines.extend(
            naturalize_interact_held_change(
                t=t,
                agent_id=agent_id,
                held_before=p0.get("held"),
                held_after=p1.get("held"),
                before_objects=before_objects,
                after_objects=after_objects,
                delivered=delivered,
                soup_change_location=soup_change_location,
                tracker=tracker,
            )
        )

    # Goal event line is allowed because it is not an internal object state diff.
    # It helps the LLM decide goal-reaching vs non-goal-reaching dependencies.
    if delivered:
        lines.append(f"t={t}: an Overcooked soup delivery was completed.")

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)

    return deduped


# -----------------------------
# Trajectory loading/compression
# -----------------------------


def get_episode_arrays(traj: dict[str, Any]) -> tuple[list[Any], list[Any], list[float]]:
    """Read first episode from the common Biswas-style JSON format."""
    states = traj.get("ep_states")
    actions = traj.get("ep_actions")
    rewards = traj.get("ep_rewards")

    if not states or not rewards:
        raise ValueError("Trajectory must contain ep_states and ep_rewards.")

    if not isinstance(states, list) or not isinstance(states[0], list):
        raise ValueError("Expected ep_states to have shape [episodes][timesteps].")

    if not isinstance(rewards, list) or not isinstance(rewards[0], list):
        raise ValueError("Expected ep_rewards to have shape [episodes][timesteps].")

    if actions is None:
        # Missing actions means we cannot enforce interact-only extraction.
        # Return None actions; describe_interact_diff will emit no agent events.
        actions0 = [None for _ in rewards[0]]
    else:
        if not isinstance(actions, list) or not isinstance(actions[0], list):
            raise ValueError("Expected ep_actions to have shape [episodes][timesteps].")
        actions0 = actions[0]

    return states[0], actions0, [float(x) for x in rewards[0]]


def compress_trajectory(path: Path) -> tuple[float, list[str], list[str], list[str]]:
    """Convert one trajectory into local task reward and interact-only event lines."""
    with path.open("r", encoding="utf-8") as f:
        traj = json.load(f)

    if not isinstance(traj, dict):
        raise ValueError(
            "Expected Biswas-style dict JSON with ep_states/ep_actions/ep_rewards. "
            "If you have the old alternating-list dump, convert it first."
        )

    states, actions, rewards = get_episode_arrays(traj)
    task_rew = float(sum(rewards))
    lines: list[str] = []
    tracker = OnionSoupTracker()

    # Need state_t and state_{t+1}. The final transition is skipped if no next
    # state exists.
    n = min(len(rewards), len(actions), len(states) - 1)

    for t in range(n):
        if not isinstance(states[t], dict) or not isinstance(states[t + 1], dict):
            continue
        lines.extend(
            describe_interact_diff(
                t=t,
                s0=states[t],
                actions_t=actions[t],
                reward=rewards[t],
                s1=states[t + 1],
                tracker=tracker,
            )
        )

    return task_rew, lines, tracker.onion_summary_lines(), tracker.soup_summary_lines()


# -----------------------------
# LLM call and outputs
# -----------------------------


def build_payload(path: Path, lines: list[str], onion_summary: list[str], soup_summary: list[str]) -> dict[str, Any]:
    return {
        "trajectory_file": str(path),
        "event_filtering": (
            "Only Overcooked INTERACT-derived agent events are listed. Movement/stay events, raw actions, "
            "numeric rewards, object internal states, and object-map state changes were removed. "
            "Onion and soup IDs were assigned by deterministic lineage tracking."
        ),
        "task_goal_hint": (
            "This is Overcooked. Two agents cooperate to make onion soups and deliver them. "
            "Useful cooperation often involves one agent making a specific onion ID, dish, or soup ID available "
            "and the other agent using that exact resource later. Goal-reaching chains should end in an Overcooked soup delivery."
        ),
        "onion_id_summary": onion_summary,
        "soup_composition_summary": soup_summary,
        "natural_language_interact_events": lines,
    }


def call_llm(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
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

    return json.loads(response.output_text)


def classify_episode(client: Any, path: Path) -> dict[str, Any]:
    task_rew, lines, onion_summary, soup_summary = compress_trajectory(path)
    payload = build_payload(path, lines, onion_summary, soup_summary)

    result = call_llm(client, payload)

    # Add local bookkeeping after the model response.
    result["task_rew"] = task_rew
    result["trajectory_file"] = str(path)
    result["num_lines_sent"] = len(lines)
    result["num_onions_tracked"] = len(onion_summary)
    result["num_soups_tracked"] = len(soup_summary)

    return result


def write_prompt_preview(path: Path, preview_dir: Path) -> None:
    task_rew, lines, onion_summary, soup_summary = compress_trajectory(path)
    payload = build_payload(path, lines, onion_summary, soup_summary)
    payload["computed_task_rew_not_sent_for_counting"] = task_rew

    preview_dir.mkdir(parents=True, exist_ok=True)
    out_path = preview_dir / f"{path.stem}.prompt.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight interact-only LLM interdependence evaluator for Overcooked trajectory JSONs."
    )
    parser.add_argument("--traj_dir", required=True, help="Directory containing trajectory .json files.")
    parser.add_argument("--out_csv", default="obj_results_llm.csv", help="Output CSV path.")
    parser.add_argument(
        "--out_jsonl",
        default="llm_episode_results.jsonl",
        help="Detailed JSONL output path with one object per trajectory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N trajectories.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not call the API; write prompt previews and print summary instead.",
    )
    parser.add_argument(
        "--preview_dir",
        default="llm_prompt_previews",
        help="Directory for --dry_run prompt preview JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    traj_paths = sorted(Path(args.traj_dir).glob("*.json"))
    if args.limit is not None:
        traj_paths = traj_paths[: args.limit]

    if not traj_paths:
        raise FileNotFoundError(f"No .json trajectories found in {args.traj_dir}")

    if args.dry_run:
        preview_dir = Path(args.preview_dir)
        for path in tqdm(traj_paths, desc="Writing prompt previews"):
            write_prompt_preview(path, preview_dir)
            task_rew, lines, onion_summary, soup_summary = compress_trajectory(path)
            print(f"\n{path}")
            print(f"  local task_rew={task_rew}")
            print(f"  interact event lines={len(lines)}")
            print(f"  onions tracked={len(onion_summary)}")
            print(f"  soups tracked={len(soup_summary)}")
            if onion_summary:
                print("  Onion ID summary:")
                for line in onion_summary[:12]:
                    print("   ", line)
                if len(onion_summary) > 12:
                    print(f"    ... {len(onion_summary) - 12} more onions")
            if soup_summary:
                print("  Soup composition summary:")
                for line in soup_summary:
                    print("   ", line)
            print("  Event lines:")
            for line in lines:
                print("   ", line)
        print(f"\nPrompt previews written to {preview_dir}")
        return

    from openai import OpenAI

    client = OpenAI()
    rows: list[list[Any]] = []

    with open(args.out_jsonl, "w", encoding="utf-8") as jf:
        for path in tqdm(traj_paths, desc="Classifying trajectories"):
            result = classify_episode(client, path)
            jf.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Match detect_int_proxy.py CSV shape:
            # [task_rew, cons_int, non_cons_int, loop_int, irr_int]
            rows.append(
                [
                    result["task_rew"],
                    result["cons_int"],
                    result["non_cons_int"],
                    result["loop_int"],
                    result["irr_int"],
                ]
            )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerows(rows)

    if rows:
        means = [sum(float(row[i]) for row in rows) / len(rows) for i in range(5)]
        print("Wrote:", args.out_csv)
        print("Wrote:", args.out_jsonl)
        print("N:", len(rows))
        print("[task_rew, cons_int, non_cons_int, loop_int, irr_int] means:")
        print([round(x, 3) for x in means])


if __name__ == "__main__":
    main()
