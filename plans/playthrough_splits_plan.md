# Playthrough splits and access policy

Written for: whoever implements this (you, or me in a later session). Assumes familiarity with
`execution/strategist/`, `run_playthrough.py` and `plans/prime_agent_sandbox_plan.md`.

## Goal

Game-level train/test splits for the playthrough leg, configurable per experiment, with two
access policies:

- **train** — full emulator access: state, save/load, reset. What the strategist does today.
- **test** — observation only: no state, no reset, no save/load.

A split must be nameable and swappable, so that `{train: red, crystal, brown | test: prism}`
and `{train: red, crystal | test: brown, prism}` are two configurations of one mechanism, not
two code paths.

## Why this is not the execution leg's split mechanism

`env_server/splits.py` partitions **1000 tasks** across 20 game variants with a seeded
per-game ratio. That shape is wrong here:

| | Execution leg | Playthrough leg |
|---|---|---|
| Unit | one short task | one long playthrough |
| Population | 1000 tasks | 4 games (`runs/games.sh` `POKEMON_GAMES`) |
| Split axis | task id, stratified within game | the game itself |
| Selection | seeded random | named, hand-chosen |
| Runs per unit | many, scored independently | one, resumed across jobs |

With four games there is nothing to randomise: the interesting splits are the ones a person
picks to isolate a hypothesis (contamination: official vs fan-made; or hold out the one title
with the most unusual map). So a playthrough split is a **named, hand-written partition**,
validated rather than generated.

## The blocker: the strategist is structurally RAM-coupled

"Test gives only obs" cannot be delivered by removing a key from a payload. The playthrough
leg reads RAM at seven points, and some of them feed prompts directly:

| Where | What it reads | Reaches the agent? |
|---|---|---|
| `execution/pokemon/supervisors.py:324` | `agent_state` | **yes** — interpolated into `[AGENT_STATE]` in the menu-objective prompt |
| `execution/pokemon/supervisors.py:395-400` | `agent_state` | **yes** — routes which executor handles an interruption |
| `execution/pokemon/supervisors.py:98,354` | `n_battles_lost` | **yes** — produces the `lost_battle` status, which triggers defeat reflection and writes a goal |
| `execution/strategist/pokemon/strategist.py:380` | `agent_state == FREE_ROAM` | **yes, indirectly** — gates global-map updates, and the map block goes into the dispatch prompt |
| `execution/strategist/pokemon/strategist.py:651,197` | `save_custom_state` / `load_custom_state` | no — per-episode checkpoint and resume |
| `execution/strategist/pokemon/strategist.py:345` | `_emulator.check_if_done()` | no — stop reason |
| `execution/strategist/pokemon/strategist.py:59-62` | frame + `_emulator.step(None)` | no — `settle_screen` waits out a door warp |

This also means the leg **already violates the vision-only rule** on every run, train or test.
That is worth knowing independently of this plan.

## The distinction that resolves it

Split the seven touchpoints by *who consumes the signal*:

- **Agent-facing perception** (rows 1-4) — shapes what the model is told or which code path
  runs. Test must deny these. They have to be re-derived from the screen.
- **Harness mechanics** (rows 5-7) — checkpointing, the done-check, waiting out a black
  screen. The agent never learns anything from them. Test may keep these, **executed
  server-side**, as long as the agent cannot address or trigger them.

This is the same obs / `state` / `info` firewall as the execution leg, drawn one layer up.
In particular it means "no resets" is a rule about the *agent*, not about the harness: the
server may still checkpoint so a 7-day job can resume, because the agent can neither request
a checkpoint nor name one to roll back to.

## Design

### 1. Split config — `configs/splits/playthrough_splits.yaml`

```yaml
universe: [pokemon_red, pokemon_crystal, pokemon_brown, pokemon_prism]

splits:
  holdout_prism:
    description: One fan-made title held out; the other three train.
    train: [pokemon_red, pokemon_crystal, pokemon_brown]
    test:  [pokemon_prism]

  holdout_fanmade:
    description: Contamination arm — official titles train, fan-made are held out.
    train: [pokemon_red, pokemon_crystal]
    test:  [pokemon_brown, pokemon_prism]
```

Validated on load: every split is a partition of `universe` (disjoint **and** exhaustive), every
name resolves in `POKEMON_GAMES`, and no split is empty on either side. Exhaustiveness is
deliberate — a game silently in neither side is a game whose results nobody can interpret.

New module `env_server/playthrough_splits.py`: `load(name)`, `kind_of(game, split)` →
`"train" | "test"`, `games(split, kind)`, `verify()`.

### 2. Policy — `env_server/playthrough_policy.py`

```python
@dataclass(frozen=True)
class PlaythroughPolicy:
    kind: str              # train | test
    state: bool            # get_info() beyond the screen
    agent_reset: bool      # the agent may reset
    agent_save_load: bool  # the agent may checkpoint and roll back
    harness_checkpoint: bool  # the server may, invisibly
```

- `train` → `state=True, agent_reset=True, agent_save_load=True, harness_checkpoint=True`
- `test` → `state=False, agent_reset=False, agent_save_load=False, harness_checkpoint=True`

Resolved from `(game, split)` at launch, not per request.

### 3. Enforcement — a wrapper, not a server

The execution leg enforces over HTTP because its agent is prime-agent in a sandbox. The
playthrough agent is our own in-process code, so the enforcement point is a wrapper around
`Environment` that the strategist is handed instead of the real one:

```
run_playthrough.py
  --split holdout_prism
      -> resolve kind for --game
      -> PolicyEnvironment(real_env, policy)   # refuses what the policy forbids
      -> PokemonStrategist(env=PolicyEnvironment, senses=...)
```

`PolicyEnvironment` raises on a forbidden call rather than returning something empty — a
silent empty dict would be indistinguishable from a game state and would corrupt a run rather
than stop it. The same `PlaythroughPolicy` can later back an HTTP server if the playthrough
agent is ever sandboxed; the policy object is the reusable part, the transport is not.

Note `MappingEnvironment` (`global_map.py:1157`) already wraps the env and calls `get_info()`.
It must wrap the `PolicyEnvironment`, not the raw one, or it is a hole straight through.

### 4. Perception shim — the real work

```python
class Senses(Protocol):
    def agent_state(self) -> AgentState: ...
    def battles_lost(self) -> int: ...
    def free_roam(self) -> bool: ...
```

- `StateSenses` — reads `get_info()`. Exactly today's behaviour. Train only.
- `VisionSenses` — derives the same three from the screen. Test only.

`agent_state` is the hard one: four-way between free roam, dialogue, battle and menu. Two
candidate implementations, to be decided by measurement, not by argument:

- **Tile-based.** The `TileRecognizer` already classifies the screen and the supervisor
  already computes `cell_kinds`. A dialogue box and a battle screen have distinctive tile
  layouts. Cheap, deterministic, no extra VLM call per step.
- **VLM classifier.** One small call per step. Costly at 200 steps/episode × 400 episodes,
  and it puts a second model in the loop.

Start tile-based and measure against `StateSenses` on a train game — that comparison is free
and exact, because both can run on the same frames. Ship whichever clears a threshold you set
in advance.

`battles_lost` is easier: the blackout sequence is a distinctive screen. `free_roam` falls
out of `agent_state`.

### 5. What test mode costs, and the control you need

Train runs with RAM senses and test runs with vision senses differ in **two** ways at once:
the game is unseen *and* the perception is degraded. A gap between them does not attribute to
either cause.

Recommended: make `--senses {state,vision}` an explicit flag, independent of the split, and
run train games under both. That gives:

| | state senses | vision senses |
|---|---|---|
| train games | today's baseline | perception cost, isolated |
| test games | generalisation cost, isolated | the headline number |

Three cells to fill instead of two, and the headline number becomes decomposable. Without the
off-diagonal cells the main result is confounded.

### 6. Provenance and artifacts

`_run_config()` gains `split`, `split_kind` and `senses`. A run directory must not mix
policies: refuse to resume when the recorded `split_kind` or `senses` differs from the current
invocation, the same way `start_episode` already refuses to overwrite an existing episode.

## Phases

1. **Splits + policy + verification.** Config, loader, `PlaythroughPolicy`, `verify()`. No
   behaviour change; `--split` optional and absent means today's full access.
2. **`PolicyEnvironment`.** Enforce on train (where it should be a no-op) and prove it is a
   no-op by re-running a short playthrough and diffing the episode records.
3. **`Senses` extraction.** Pull the seven touchpoints behind `StateSenses`. Still no
   behaviour change — this is the refactor that makes step 4 possible.
4. **`VisionSenses`.** Tile-based first. Measure agreement with `StateSenses` frame by frame
   on an existing run's archived frames (`report_detail=executor` runs already hold them, so
   this needs no new emulator time).
5. **Harness-side checkpointing.** Move `save_custom_state` out of the agent's reach so test
   runs resume without the agent being able to.
6. **Wire up `run_playthrough.py`,** the wrapper script and the slurm scripts.

Steps 1-3 are mechanical and safe. Step 4 is the one with real risk and should be gated on the
agreement measurement before anything is run at scale.

## Open questions

1. **Is the playthrough agent ever going to be sandboxed prime-agent**, or does it stay as
   `PokemonStrategist`? If sandboxed, step 3's `Senses` boundary becomes an HTTP boundary and
   the design above still holds, but the transport work from the execution leg gets reused.
2. **Does `settle_screen` count as an action?** It steps the emulator with no button. I have
   it as harness mechanics, but it does advance game time, so an argument exists that a test
   agent should pay for those frames out of its budget.
3. **Should test allow the one implicit reset at boot?** `run_playthrough.py:117` calls
   `environment.reset()` once before the strategist starts. Read as "the game is switched on"
   it is not a reset at all. The plan assumes it is allowed exactly once and refused
   thereafter.
