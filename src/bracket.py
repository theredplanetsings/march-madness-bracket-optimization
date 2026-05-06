"""Bracket structure, simulation, and ESPN scoring.

A bracket is 4 regions × 16 seeds = 64 teams. Round-1 seed pairings within
each region are fixed: (1,16), (8,9), (5,12), (4,13), (6,11), (3,14),
(7,10), (2,15). Region winners advance to the Final Four, paired
according to the official NCAA region pairing for that year.

We expose:
  - Bracket: dict of {region: [team@seed1, ..., team@seed16]} plus a
    Final Four pairing of the 4 regions.
  - simulate_once(bracket, win_prob_fn, rng) -> dict mapping each team to
    their max round reached.
  - simulate_many(bracket, win_prob_fn, n_sims) -> DataFrame of per-team
    round-reach probabilities.
  - espn_score(picks, outcome) -> int total points.

ESPN Tournament Challenge scoring: 10/20/40/80/160/320 per correct pick.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd

# Round-1 pairings inside a region (top-half then bottom-half of bracket)
ROUND1_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "NCG"]
ESPN_POINTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320}


@dataclass
class Bracket:
    """The set of 64 teams arranged into a tournament bracket.

    regions: dict region_name -> list[16] team names indexed by seed-1 (so
             regions[r][0] is the #1 seed, regions[r][15] is the #16 seed).
    final_four_pairs: list of two tuples of region names. The winners of
                      each pair play for the championship.
    """
    regions: dict[str, list[str]]
    final_four_pairs: list[tuple[str, str]]

    @property
    def all_teams(self) -> list[str]:
        out = []
        for teams in self.regions.values():
            out.extend(teams)
        return out


def build_prob_matrix(teams: list[str], win_prob: Callable[[str, str], float]) -> np.ndarray:
    """Precompute the 64x64 win-probability matrix P[i,j] = P(team i beats team j)."""
    n = len(teams)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i, j] = win_prob(teams[i], teams[j])
    return P


def simulate_many(bracket: Bracket, win_prob: Callable[[str, str], float],
                  n_sims: int = 10000, seed: int = 7) -> pd.DataFrame:
    """Vectorized simulation. Precompute the 64x64 win-prob matrix once;
    each simulation is then table lookups + RNG draws (fast). Returns a
    DataFrame indexed by team with columns p_R64..p_Champ."""
    teams = bracket.all_teams
    n = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    P = build_prob_matrix(teams, win_prob)

    # Bracket structure as integer indices into `teams`.
    # regions: in fixed order, each region is 16 consecutive teams.
    region_names = list(bracket.regions.keys())
    region_team_idx: list[list[int]] = []  # list of length 4, each [16] indices
    for r in region_names:
        idxs = [team_to_idx[t] for t in bracket.regions[r]]
        region_team_idx.append(idxs)
    # Final Four pairing as region indices
    f4_pair_idx = [(region_names.index(a), region_names.index(b)) for a, b in bracket.final_four_pairs]

    rng = np.random.default_rng(seed)

    # Counts: per team, count of sims where they reached at least round r.
    # round indices: 0=R64 (always 1.0), 1=R32, 2=S16, 3=E8, 4=F4, 5=NCG, 6=Champ
    reach_counts = np.zeros((n, 7), dtype=np.int64)

    # Pre-draw all randoms for speed: 63 games per sim
    # Layout: 4 regions * 15 games + 2 F4 + 1 NCG = 63
    randoms = rng.random((n_sims, 63))

    pair_layout = list(ROUND1_PAIRS)
    for s in range(n_sims):
        r_idx = 0
        region_winners = []
        for region in region_team_idx:
            # Round 1
            r1 = []
            for (s1, s2) in pair_layout:
                a, b = region[s1 - 1], region[s2 - 1]
                w = a if randoms[s, r_idx] < P[a, b] else b
                r1.append(w); r_idx += 1
                reach_counts[w, 1] += 1
            # Round 2
            r2 = []
            for i in range(0, 8, 2):
                a, b = r1[i], r1[i + 1]
                w = a if randoms[s, r_idx] < P[a, b] else b
                r2.append(w); r_idx += 1
                reach_counts[w, 2] += 1
            # Sweet 16
            r3 = []
            for i in range(0, 4, 2):
                a, b = r2[i], r2[i + 1]
                w = a if randoms[s, r_idx] < P[a, b] else b
                r3.append(w); r_idx += 1
                reach_counts[w, 3] += 1
            # Elite 8
            a, b = r3[0], r3[1]
            rw = a if randoms[s, r_idx] < P[a, b] else b
            r_idx += 1
            reach_counts[rw, 4] += 1
            region_winners.append(rw)
        # Final Four
        f4_winners = []
        for (i_r1, i_r2) in f4_pair_idx:
            a, b = region_winners[i_r1], region_winners[i_r2]
            w = a if randoms[s, r_idx] < P[a, b] else b
            r_idx += 1
            reach_counts[w, 5] += 1
            f4_winners.append(w)
        # Championship
        a, b = f4_winners
        champ = a if randoms[s, r_idx] < P[a, b] else b
        reach_counts[champ, 6] += 1

    # All teams reached R64 trivially
    reach_counts[:, 0] = n_sims

    out = pd.DataFrame(reach_counts / n_sims, index=teams,
                       columns=["p_R64", "p_R32", "p_S16", "p_E8", "p_F4", "p_NCG", "p_Champ"])
    out.index.name = "team"
    return out


def simulate_outcomes(bracket: Bracket, win_prob: Callable[[str, str], float],
                      n_sims: int = 10000, seed: int = 7) -> np.ndarray:
    """Like simulate_many but returns the raw advancement matrix (n_sims, 64)
    of max-round-reached per team. Needed for pool-percentile analysis where
    we score every simulation against every candidate strategy."""
    teams = bracket.all_teams
    n = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    P = build_prob_matrix(teams, win_prob)

    region_names = list(bracket.regions.keys())
    region_team_idx = [[team_to_idx[t] for t in bracket.regions[r]] for r in region_names]
    f4_pair_idx = [(region_names.index(a), region_names.index(b)) for a, b in bracket.final_four_pairs]

    rng = np.random.default_rng(seed)
    randoms = rng.random((n_sims, 63))
    advancements = np.zeros((n_sims, n), dtype=np.int8)

    pair_layout = list(ROUND1_PAIRS)
    for s in range(n_sims):
        r_idx = 0
        region_winners = []
        for region in region_team_idx:
            r1 = []
            for (s1, s2) in pair_layout:
                a, b = region[s1 - 1], region[s2 - 1]
                w = a if randoms[s, r_idx] < P[a, b] else b
                r1.append(w); r_idx += 1
                advancements[s, w] = max(advancements[s, w], 1)
            r2 = []
            for i in range(0, 8, 2):
                a, b = r1[i], r1[i + 1]
                w = a if randoms[s, r_idx] < P[a, b] else b
                r2.append(w); r_idx += 1
                advancements[s, w] = max(advancements[s, w], 2)
            r3 = []
            for i in range(0, 4, 2):
                a, b = r2[i], r2[i + 1]
                w = a if randoms[s, r_idx] < P[a, b] else b
                r3.append(w); r_idx += 1
                advancements[s, w] = max(advancements[s, w], 3)
            a, b = r3[0], r3[1]
            rw = a if randoms[s, r_idx] < P[a, b] else b
            r_idx += 1
            advancements[s, rw] = max(advancements[s, rw], 4)
            region_winners.append(rw)
        f4_winners = []
        for (i_r1, i_r2) in f4_pair_idx:
            a, b = region_winners[i_r1], region_winners[i_r2]
            w = a if randoms[s, r_idx] < P[a, b] else b
            r_idx += 1
            advancements[s, w] = max(advancements[s, w], 5)
            f4_winners.append(w)
        a, b = f4_winners
        champ = a if randoms[s, r_idx] < P[a, b] else b
        advancements[s, champ] = 6

    return advancements, teams, P


def espn_score(picks: dict[int, dict[str, str]], actual: dict[str, int]) -> int:
    """Score a bracket against actual outcomes.

    picks: {round_idx (1..6): {slot_id: predicted_team}} where slot_id
        identifies the bracket position. We keep it simple: we just count
        a pick correct if the predicted team actually reached at least
        that round.
    actual: {team: max_round_index_reached (0=R64..6=Champion)}.

    A correct pick at round R means: the team you picked to *be at round
    R+1* (i.e. *win* round R) actually did win that round, i.e. reached
    round R+1 (advancement value >= R+1).

    We use a simpler representation here: picks is {round_name:
    set_of_teams} = the set of teams you picked to advance through that
    round. For ESPN scoring we award points for each team in that set
    whose actual advancement is at least the round index.
    """
    total = 0
    round_idx = {n: i for i, n in enumerate(["R32", "S16", "E8", "F4", "NCG", "Champ"], start=1)}
    for rn, teams_picked in picks.items():
        ri = round_idx[rn]
        pts = ESPN_POINTS["R64"] if rn == "R32" else ESPN_POINTS[
            {"R32": "R32", "S16": "S16", "E8": "E8", "F4": "F4", "NCG": "NCG", "Champ": "NCG"}[rn]
        ]
        # round_name in picks denotes "team picked to reach this round (i.e. win previous round)"
        # ESPN scoring is by round won. R32 = 10pts/correct (winning R64), etc.
        round_to_points = {"R32": 10, "S16": 20, "E8": 40, "F4": 80, "NCG": 160, "Champ": 320}
        pts = round_to_points[rn]
        for t in teams_picked:
            if actual.get(t, 0) >= ri:
                total += pts
    return total
