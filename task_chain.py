"""The real multi-task metric: a static chain of go-to subgoals scored out of N.

For now the chain is a fixed list (stand-in for the LLM that will eventually
string subgoals together); score = how many legs the navigator reaches in one
episode, pose persisting leg-to-leg. Top score = len(chain).

Shared by the in-training TB metric (agent.chain_eval) and the offline harness
so both report the identical number.

3D task model (homebot3d.tasks.TaskManager):
  - trash: each pile is collected by driving within REACH_RADIUS of it.
  - drink (carry goal): pick up at the fridge, deliver to the seated human
    (recliner).
  - package (carry goal): pick up at the door, deliver to the human (recliner).
The TaskManager auto-advances phases whenever the robot is close enough during a
step, so the chain just navigates to each leg's coordinate and reads the honest
world-state delta to decide whether the leg accomplished anything.

The 5-leg chain mirrors the old 2D chain
(trash -> fridge -> human -> door -> human): collect trash, fetch the drink from
the fridge, deliver it to the human, fetch the parcel from the door, deliver it
to the human.
"""

# trash >> pick up drink (fridge) >> deliver drink (human) >>
# pick up package (door) >> deliver package (human)
DEFAULT_CHAIN = ["trash", "drink_pickup", "drink_deliver",
                 "package_pickup", "package_deliver"]


def world_state(base) -> dict:
    """Snapshot the task-relevant world state -- the honest ground truth for
    whether a leg accomplished anything. Reads homebot3d's TaskManager, whose
    get_info() reports trash_remaining, the carrying set, and per-goal phase."""
    info = base._tasks.get_info(base._robot)
    carrying = set(info["carrying"])
    return {
        "carrying": carrying,
        "trash_remaining": info["trash_remaining"],
        "drink_phase": info["drink_phase"],      # seek_source | seek_target | done
        "package_phase": info["package_phase"],
        "robot_xy": (float(base._robot.x), float(base._robot.y)),
    }


def leg_succeeded(name, before, after, arrived) -> bool:
    """Honest leg success: did the task actually happen (a world-state delta) --
    not merely 'the robot got near the coordinate'. `arrived` is the fallback
    only for pure-navigation legs that have no task effect."""
    if name == "trash":
        return after["trash_remaining"] < before["trash_remaining"]
    if name == "drink_pickup":
        # drink entered the carry set (phase advanced seek_source -> seek_target).
        return ("drink" in after["carrying"]) and ("drink" not in before["carrying"])
    if name == "drink_deliver":
        return after["drink_phase"] == "done" and before["drink_phase"] != "done"
    if name == "package_pickup":
        return ("package" in after["carrying"]) and ("package" not in before["carrying"])
    if name == "package_deliver":
        return after["package_phase"] == "done" and before["package_phase"] != "done"
    return arrived


def resolve_goal(base, name):
    """Map a chain subgoal name to world (x, y) metres. Resolve all legs up front
    (before stepping) so incidental trash pickup can't empty trash_positions
    mid-chain, and so a carry goal's source tile is captured before the phase
    auto-advances."""
    from homebot3d.world import tile_center

    tasks = base._tasks
    map_ = base._map
    if name == "trash":
        # The first spawned pile; if trash is already empty, fall back to the tile.
        if tasks.trash_positions:
            return tile_center(*tasks.trash_positions[0])
        return (float(base._robot.x), float(base._robot.y))
    if name == "drink_pickup":
        return tile_center(*map_.pickup_tiles["drink"])
    if name == "drink_deliver":
        return tile_center(*map_.dropoff_tiles["drink"])
    if name == "package_pickup":
        return tile_center(*map_.pickup_tiles["package"])
    if name == "package_deliver":
        return tile_center(*map_.dropoff_tiles["package"])
    raise ValueError(f"unknown chain leg: {name}")
