#!/usr/bin/env python


"""
Code for simulating the evolution of mutation rate in haploid or diploid
populations with different modes of reproduction.
"""


# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

import argparse
import doctest
import multiprocessing as mp
import sys
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import ClassVar

import numpy as np
import numpy.random as rnd
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Set up random number generator (RNG)
# =============================================================================


# The following resets the RNG to an arbitrary state
rng: rnd.Generator = rnd.default_rng()

# The following sets the RNG to a particular state (in this case, that
# corresponding to seed=42).  This is used in testing.
#
# rng_state: dict = {
#     'bit_generator': 'PCG64',
#     'state': {
#         'state': 274674114334540486603088602300644985544,
#         'inc': 332724090758049132448979897138935081983
#     },
#     'has_uint32': 0,
#     'uinteger': 0
# }
#
# rng = rnd.Generator(rnd.PCG64())
# rng.bit_generator.state = rng_state


# =============================================================================
# Recombination
# =============================================================================


def single_point_crossover(
    seq1: npt.NDArray, seq2: npt.NDArray, pos: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Perform a single-point crossover between two sequences at a given position.
    Return both recombined sequences.

    Based on code from: https://tinyurl.com/ymrfcjy7

    Parameters
    ----------
    seq1 : npt.NDArray
        The first sequence.
    seq2 : npt.NDArray
        The second sequence.
    pos : int
        The position at which the crossover occurs. Must be between 1 and len(seq1) - 1.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        Two recombined sequences.

    Example
    -------
    >>> seq1 = np.array([0, 1, 2, 3, 4])
    >>> seq2 = np.array([5, 6, 7, 8, 9])
    >>> pos = 2
    >>> rec1, rec2 = single_point_crossover(seq1, seq2, pos)
    >>> print(rec1)
    [0 1 7 8 9]
    >>> print(rec2)
    [5 6 2 3 4]
    >>> seq1, seq2 = single_point_crossover(rec1, rec2, pos)
    >>> print(seq1)
    [0 1 2 3 4]
    >>> print(seq2)
    [5 6 7 8 9]
    """
    assert len(seq1) == len(seq2), "Sequences must have the same length."
    assert 0 < pos < len(seq1), "Crossover position must be between 1 and L–1."
    rec1: npt.NDArray = np.concatenate((seq1[:pos], seq2[pos:]))
    rec2: npt.NDArray = np.concatenate((seq2[:pos], seq1[pos:]))
    return rec1, rec2


def multi_point_crossover(
    seq1: npt.NDArray, seq2: npt.NDArray, pos: list[int] | npt.NDArray[np.int_]
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Perform a multi-point crossover operation between two sequences at multiple
    specified positions.  Return both recombined sequences

    Based on code from: https://tinyurl.com/ymrfcjy7

    Parameters
    ---------
    seq1 : npt.NDArray
        The first sequence.
    seq2 : npt.NDArray
        The second sequence.
    pos : list[int]
        A list of positions at which crossovers occur. Each position must be between 1 and len(seq1) - 1.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        Two recombined sequences.

    Example
    -------
    >>> seq1 = np.array([0, 1, 2, 3, 4])
    >>> seq2 = np.array([5, 6, 7, 8, 9])
    >>> pos = [3, 1, 1]
    >>> rec1, rec2 = multi_point_crossover(seq1, seq2, pos)
    >>> print(rec1)
    [0 6 7 3 4]
    >>> print(rec2)
    [5 1 2 8 9]
    """
    upos: npt.NDArray[np.int_] = np.unique(pos)  # Ensure positions are sorted and unique
    for i in upos:
        seq1, seq2 = single_point_crossover(seq1, seq2, i)
    return seq1, seq2


def meiosis(
    diploids: npt.NDArray[np.int_], i: npt.NDArray[np.int_], rmap: list[float]
) -> npt.NDArray[np.int_]:
    """
    Generate haploids from diploids.

    Parameters
    ----------
    diploids : npt.NDArray[np.int_]
        Diploids.
    i : npt.NDArray[np.int_]
        Indices of diploid individuals to pick to produce haploids.
    rmap : list[float]
        Recombination frequencies between adjacent loci.

    Returns
    -------
    npt.NDArray[np.int_]
        Haploids.
    """
    L: int = len(diploids[0, 0, :])
    assert len(rmap) == L - 1
    n: int = len(i)
    if rmap == [0.5] * (L - 1):
        # free recombination
        p: npt.NDArray[np.float64] = diploids.sum(axis=0) / 2
        haploids: npt.NDArray = rng.binomial(1, p[i, :])
    else:
        # segregation
        j: npt.NDArray = rng.integers(low=0, high=1, size=n, endpoint=True)
        haploids = diploids[j, i, :]
        if rmap != [0.0] * (L - 1):
            # crossovers if rmap > 0
            crossovers: npt.NDArray = rng.binomial(1, rmap, size=(n, L - 1))
            for k in np.where(crossovers.sum(axis=1) > 0)[0]:
                pos: npt.NDArray = np.where(crossovers[k] == 1)[0] + 1
                haploids[k] = multi_point_crossover(diploids[0][i[k]], diploids[1][i[k]], pos)[j[k]]
    return haploids


# =============================================================================
# Simulation
# =============================================================================


@dataclass(kw_only=True)
class Population(ABC):
    """
    Population with loci controlling fitness and mutation rate.

    Parameters
    ----------
    mode : str
        Mode of reproduction.
    ancestor : str
        Type of ancestral population.
    N : int
        Population size.
    loci_w : list[int]
        Fitness loci.
    loci_m : list[int]
        Mutator loci.
    loci_a : list[int]
        Antimutator loci.
    base_w : float
        Baseline mutation rate of fitness loci.
    base_m : float
        Baseline mutation rate of mutator loci.
    base_a : float
        Baseline mutation rate of antimutator loci.
    s_w : float
        Deleterious effect of mutations in fitness loci.
    s_m : float
        Effect of mutations in mutator loci.
    s_a : float
        Effect of mutations in antimutator loci.
    r : float | list[float]
        Recombination frequency between adjacent loci. If a single number is
        given, the values are equal between every pair of adjacent loci.
        Default = 0.
    tau : None | int | list[int], optional (required if mode='fac_*')
        Number of generations between sexual events. Default = None.
    sync : None | bool, optional (required if mode='fac_*')
        Whether facultative sex is synchronous. Default = None.
    """

    ploidy: int = field(init=False)
    mode: str
    ancestor: str
    N: int
    t: int = 0
    loci_w: list[int]
    loci_m: list[int]
    loci_a: list[int]
    base_w: float
    base_m: float
    base_a: float
    s_w: float
    s_m: float
    s_a: float
    r: float | list[float] = 0.0
    tau: None | int | list[int] = None
    sync: None | bool = None

    def __post_init__(self) -> None:
        """
        Finish initializing population:

        - Define L
        - Validate:
            - reproductive mode
            - genome parameters
            - r, tau, and sync
        - Generate population
        """
        self._validate_mode()
        self._validate_genome_pars()
        self._validate_r()
        self._validate_fac()
        self.generate_population()

    def _validate_genome_pars(self) -> None:
        """Check validity of genome parameters."""
        # loci_*
        self.L = len(self.loci_w) + len(self.loci_m) + len(self.loci_a)
        assert self.L > 0, "Number of loci must be greater than 0."
        for loci in [self.loci_w, self.loci_m, self.loci_a]:
            assert len(loci) >= 0, "Number of loci cannot be negative."
        assert sum(self.loci_w) + sum(self.loci_m) + sum(self.loci_a) == np.arange(self.L).sum(), (
            "Loci must be numbered continuously between 0 and L-1 where L is the total number of loci."
        )
        # base_*
        for base in [self.base_w, self.base_m, self.base_a]:
            assert base >= 0, "Baseline mutation rates must be positive."
        # s_*
        assert 0 <= self.s_w <= 1, "s_w must be between 0 and 1."
        assert self.s_m >= 0, "s_m must be positive."
        assert 0 <= self.s_a <= 1, "s_w must be between 0 and 1."

    @abstractmethod
    def _validate_mode(self) -> None:
        """Check validity of reproductive mode."""

    def _validate_r(self) -> None:
        """Check validity of r.  Generate recombination map."""
        if isinstance(self.r, (float, int)):
            self.rmap: list[float] = [float(self.r)] * (self.L - 1)
        elif isinstance(self.r, list):
            self.rmap = self.r
        assert len(self.rmap) == self.L - 1, (
            "Length of recombination map must equal number of loci - 1."
        )
        assert max(self.rmap) <= 0.5, "r between adjacent loci cannot exceed 0.5."
        assert min(self.rmap) >= 0.0, "r between adjacent loci cannot be negative."
        if self.mode == "asex":
            assert self.nchroms == 1, f"nchroms = {self.nchroms} but must be = 1 in {self.mode}."
            assert max(self.rmap) == 0.0, (
                f"r must be zero between all pairs of adjacent loci in {self.mode}."
            )

    @staticmethod
    def get_genetic_map(
        *,
        nloci: tuple[int, int, int],
        r: float,
        w_only: None | bool = None,
        u_only: None | bool = None,
        nchroms: None | int = None,
    ) -> tuple[list[int], list[int], list[int], list[float]]:
        """
        Generate locus positions and recombimation map.

        Used in quick_init() methods.

        w_only and u_only suppress recombination between certain loci.

        nchroms causes random chromosome splitting.

        Parameters
        ----------
        nloci : tuple[int]
            Number of loci of each kind.
        r : float
            Recombination frequency between adjacent loci.
        w_only : bool, optional
            Only allow crossovers between fitness loci.  Suppress recombination
            between mutation rate loci.  Loci positions not shuffled.  Default = None.
        u_only : bool, optional
            Only allow crossovers between mutation rate loci.  Suppress recombination
            between fitness loci.  Mutation rate loci positions shuffled; fitness
            loci positions not shuffled.  Default = None.
        nchroms : int, optional
            Number of chromosomes to be generated.  Default = None.

        Returns
        -------
        tuple[list[int], list[int], list[int], list[float]]
            Loci positions and recombination map.

        Examples
        --------
        >>> loci_w, loci_m, loci_a, rmap = Population.get_genetic_map(nloci=(3, 2, 2), r=0.5)
        >>> rmap
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        >>> loci_w, loci_m, loci_a, rmap = Population.get_genetic_map(nloci=(3, 2, 2), r=0.5, w_only=True)
        >>> rmap
        [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        >>> loci_m
        [3, 4]
        >>> loci_w, loci_m, loci_a, rmap = Population.get_genetic_map(nloci=(3, 2, 2), r=0.5, u_only=True)
        >>> rmap
        [0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        >>> loci_w
        [0, 1, 2]
        """
        L: int = sum(nloci)
        cumloci: npt.NDArray[np.int_] = np.cumsum(nloci)
        loci: npt.NDArray[np.int_] = np.arange(L)
        assert not (w_only and u_only), "w_only and u_only cannot both be True."
        if not (w_only or u_only):
            # shuffle all loci
            rng.shuffle(loci)
        elif u_only:
            # shuffle only mutation rate loci
            rng.shuffle(loci[nloci[0] :])
        loci_w: list[int] = loci[: nloci[0]].tolist()
        loci_m: list[int] = loci[nloci[0] : cumloci[1]].tolist()
        loci_a: list[int] = loci[cumloci[1] : cumloci[2]].tolist()
        assert isinstance(r, (int, float)) and (0 <= r <= 0.5), (
            "r must be a value between 0 and 0.5."
        )
        rmap: list[float] = [float(r)] * (L - 1)
        if w_only:
            rmap[nloci[0] :] = [0.0] * (nloci[1] + nloci[2] - 1)
        elif u_only:
            rmap[: (nloci[0] - 1)] = [0.0] * (nloci[0] - 1)
        if nchroms is not None:
            # validate nchroms
            assert L >= 4, "nchroms can only be set if L >= 4."
            assert not (w_only and u_only), (
                "nchroms can only be set if w_only and u_only are false."
            )
            assert r != 0.5, "nchroms can only be set if 0 <= r < 0.5."
            assert nchroms <= L // 2, f"nchroms can only be set up to a maximum of {L // 2}."
            interloci: npt.NDArray[np.int_] = np.arange(L - 1)
            rng.shuffle(interloci)
            chrom_breaks = interloci[: (nchroms - 1)]
            for chrom_break in chrom_breaks:
                rmap[chrom_break] = 0.5
        return loci_w, loci_m, loci_a, rmap

    def _validate_fac(self) -> None:
        """
        Check validity of parameters used in fac_sex and fac_self:
        - tau
        - sync
        """
        if self.mode[:3] == "fac":
            self.nevents: int = 0
            assert isinstance(self.tau, (int, list)), f"tau must be set for {self.mode}."
            if isinstance(self.tau, int):
                assert self.tau > 1, f"tau must be > 1 for {self.mode}."
            elif isinstance(self.tau, list):
                assert self.sync, f"tau must be a single value under asynchronous {self.mode}."
                for tau in self.tau:
                    assert tau > 0, f"Every tau must be > 0 for {self.mode}."
            assert isinstance(self.sync, bool), f"sync must be set for {self.mode}."
        else:
            assert self.tau is None, f"tau cannot be set for {self.mode}."
            assert self.sync is None, f"sync cannot be set for {self.mode}."

    @abstractmethod
    def generate_population(self) -> None:
        """Generate population."""

    @property
    def chrom_breaks(self) -> list[int]:
        """
        Infer chromosome breaks from recombination map.  Define a chromosome
        break as occurring when r=0.5 between adjacent loci.
        """
        return [i for i, r in enumerate(self.rmap) if r == 0.5]

    @property
    def nchroms(self) -> int:
        """Infer number of chromosomes."""
        return len(self.chrom_breaks) + 1

    @abstractmethod
    def update(self) -> None:
        """Calculate fitness and mutation rate of all individuals."""

    @abstractmethod
    def mutate(self) -> None:
        """Mutate loci with probability given by mutation rate."""

    def select(self, n: int) -> npt.NDArray[np.int_]:
        """
        Sample n genotypes with replacement with probability proportional to
        fitness.

        Parameters
        ----------
        n : int
            Number of individuals to select.

        Returns
        -------
        npt.NDArray[np.int_]
            Indices of individuals sampled.
        """
        cumw: npt.NDArray[np.float64] = np.cumsum(self.w)  # type: ignore
        rand: npt.NDArray[np.float64] = np.multiply(rng.random(n), cumw[self.N - 1])
        i: npt.NDArray[np.int_] = np.searchsorted(cumw, rand)
        return i

    @abstractmethod
    def asex(self, n: int) -> npt.NDArray[np.int_]:
        """Generate n individuals by asexual reproduction."""

    @abstractmethod
    def sex(self, n: int) -> npt.NDArray[np.int_]:
        """Generate individuals by sexual reproduction."""

    @abstractmethod
    def fac(self) -> npt.NDArray[np.int_]:
        """Generate a new population by facultative sexual reproduction."""

    @abstractmethod
    def next_gen(self) -> None:
        """
        Complete one round of the life-cycle including:
        - mutation
        - natural selection
        - reproduction
        """

    def evolve(
        self,
        ngens: int,
        every: int,
        track_progress: bool = False,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float], list[int]]:
        """
        Allow population to evolve for ngens.

        Parameters
        ----------
        ngens : int
            Number of generations for simulation.
        every : int
            Number of generations between data collection events.
            (1 means that data are collected every generation).
        track_progress : bool, optional
            Whether to show progress bar.  Default = False.

        Returns
        -------
        tuple[list[float], list[float], list[float], list[float], list[float]]
            Time series for each of the following population statistics:
            - Mean fitness
            - Mean deleterious mutation rate
            - Variance in fitness
            - Variance in deleterious mutation rate
            - Covariance between deleterious mutation rate and fitness
            - Generation numbers
        """
        mean_w: list[float] = []
        mean_u: list[float] = []
        var_w: list[float] = []
        var_u: list[float] = []
        cov_wu: list[float] = []

        def collect_data():
            mean_w.append(self.w.mean())  # type: ignore
            mean_u.append(self.u_w.mean() * len(self.loci_w) * self.ploidy)  # type: ignore
            var_w.append(self.w.var(ddof=1))  # type: ignore
            var_u.append(self.u_w.var(ddof=1) * (len(self.loci_w) * self.ploidy) ** 2)  # type: ignore
            cov_wu.append(np.cov(self.u_w, self.w)[1, 0] * len(self.loci_w) * self.ploidy)  # type: ignore

        collect_data()
        tassay: list[int] = [t for t in range(ngens + 1) if t % every == 0]
        if track_progress:
            for t in tqdm(range(1, ngens + 1)):
                self.next_gen()
                if t in tassay:
                    collect_data()
        else:
            for t in range(1, ngens + 1):
                self.next_gen()
                if t in tassay:
                    collect_data()
        return mean_w, mean_u, var_w, var_u, cov_wu, tassay

    def replicate_evolve(
        self, ngens: int, every: int, nreps: int, filename: None | str = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run an evolutionary simulation of nreps replicate populations for ngens
        generations.

        Parameters
        ----------
        ngens : int
            Number of generations.
        every : int
            Number of generations between data collection events.
            (1 means that data are collected every generation).
        nreps : int
            Number of replicate populations.
        filename : str
            File name root to save data.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Time series for each replicate in columns. Separate data frames for:
            - Mean fitness
            - Mean deleterious mutation rate
            - Variance in fitness
            - Variance in deleterious mutation rate
            - Covariance between deleterious mutation rate and fitness
        """
        times: list[int] = [t for t in range(ngens + 1) if t % every == 0]
        wdata: pd.DataFrame = pd.DataFrame({i: np.zeros(len(times)) for i in range(nreps)})
        udata: pd.DataFrame = pd.DataFrame({i: np.zeros(len(times)) for i in range(nreps)})
        vwdata: pd.DataFrame = pd.DataFrame({i: np.zeros(len(times)) for i in range(nreps)})
        vudata: pd.DataFrame = pd.DataFrame({i: np.zeros(len(times)) for i in range(nreps)})
        covdata: pd.DataFrame = pd.DataFrame({i: np.zeros(len(times)) for i in range(nreps)})
        for data in [wdata, udata, vwdata, vudata, covdata]:
            data["t"] = times
            data.set_index("t", drop=True, inplace=True)
        for i in tqdm(range(nreps)):
            pop = deepcopy(self)
            w, u, vw, vu, cov, t = pop.evolve(ngens, every)
            wdata[i] = w
            udata[i] = u
            vwdata[i] = vw
            vudata[i] = vu
            covdata[i] = cov
        if filename is not None:
            wdata.to_csv(filename + "_w.csv", float_format="%.6g")
            udata.to_csv(filename + "_u.csv", float_format="%.6g")
            vwdata.to_csv(filename + "_vw.csv", float_format="%.6g")
            vudata.to_csv(filename + "_vu.csv", float_format="%.6g")
            covdata.to_csv(filename + "_cov.csv", float_format="%.6g")
        return wdata, udata, vwdata, vudata, covdata

    @abstractmethod
    def reach(self, ptarget: float = 1.0) -> tuple[bool, int]:
        """
        Assay spread of an antimutator mutation in an unmutated population. A
        mutation is introduced at a random antimutator locus and the population
        is allowed to evolve until the mutation either reaches a certain target
        frequency p or disappears. Fitness loci are allowed to mutate but mutator
        and (other) antimutator loci are not.

        Parameters
        ----------
        ptarget : float (default = 1., fixation)
            Target frequency the antimutator mutation must reach.

        Returns
        -------
        tuple[bool, int]
            Whether the mutation reached target frequency (True) or disappeared
            (False) and the time it took to do so.
        """

    def replicate_reach(self, ptarget: float, nreps: int) -> tuple[int, list[int]]:
        """
        Run nreps spread assays.

        Parameters
        ----------
        ptarget : float (default = 1., fixation)
            Target frequency the antimutator mutation must reach.
        nreps : int
            Number of replicate populations.

        Returns
        -------
        tuple[int, list[int]]
            Number of times a population reached the target frequency and list
            containing times.
        """
        nreach: int = 0
        treach: list[int] = []
        for i in tqdm(range(nreps)):
            pop = deepcopy(self)
            reached, t = pop.reach(ptarget)
            if reached:
                nreach += 1
                treach.append(t)
        return nreach, treach


@dataclass(kw_only=True)
class Haploid(Population):
    """
    Population of haploids with loci controlling fitness and mutation rate.

    See Population.
    """

    modes: ClassVar[frozenset] = frozenset(["asex", "sex", "fac_sex"])

    def __post_init__(self) -> None:
        """Set ploidy.  Ensure that sexuals do not have r = 0."""
        self.ploidy: int = 1
        super().__post_init__()
        if self.mode != "asex":
            assert max(self.rmap) > 0.0, (
                "r cannot be zero between all pairs of adjacent loci in a haploid sexual or facultative sexual."
            )

    def _validate_mode(self) -> None:
        """Check validity of reproductive mode."""
        assert self.mode in Haploid.modes, (
            f"Reproductive mode must be one of the following: {', '.join(sorted(Haploid.modes))}."
        )

    def generate_population(self) -> None:
        """
        Generate population.

        Implemented types:
        - unmutated
        - mutated_p where p is the proportion of individuals mutated.
        - asex_t where population evolves under obligate asexual reproduction
          for t generations.
        - sex_t where population evolves under obligate sexual reproduction with
          free recombination for t generations.
        """
        unmut: npt.NDArray[np.int_] = np.zeros((self.N, self.L), dtype=int)
        self.population: npt.NDArray[np.int_] = unmut
        self.update()
        if self.ancestor == "unmutated":
            pass
        elif self.ancestor[:8] == "mutated_":
            p: float = float(self.ancestor.split("_")[1])
            assert 0 < p < 1, "Proportion of mutations must be between 0 and 1."
            self.population += rng.binomial(1, p, (self.N, self.L))
        elif self.ancestor[:4] == "asex":
            t: int = int(self.ancestor.split("_")[1])
            mode = self.mode
            self.mode = "asex"
            for i in range(t):
                self.next_gen()
            self.mode = mode
        elif self.ancestor[:3] == "sex":
            t: int = int(self.ancestor.split("_")[1])
            mode = self.mode
            rmap = self.rmap
            self.mode = "sex"
            self.rmap = [0.5] * (self.L - 1)
            for i in range(t):
                self.next_gen()
            self.mode = mode
            self.rmap = rmap
        else:
            raise NotImplementedError(f"{self.ancestor} initial state not implemented.")
        self.update()

    @staticmethod
    def quick_init(
        *,
        mode: str,
        ancestor: str,
        N: int,
        nloci: tuple[int, int, int],
        base: tuple[float, float, float],
        s: tuple[float, float, float],
        r: float = 0.0,
        tau: None | int | list[int] = None,
        sync: None | bool = None,
        w_only: None | bool = None,
        u_only: None | bool = None,
        nchroms: None | int = None,
    ) -> Haploid:
        """
        Generate a population of haploid individuals based on a simpler
        parameter specification than that of the __init__() method.

        nloci, base, and s parameters take tuples (w, s, a).

        The loci attributes are generated based on the nloci values.  The loci
        of different types are distributed randomly over a single chromosome.

        Parameters
        ----------
        mode : str
            Mode of reproduction.
        ancestor : str
            Type of ancestral population.
        N : int
            Population size.
        nloci : tuple[int]
            Number of loci of each kind.
        base : tuple[float]
            Baseline mutation rates for each kind of locus.
        s : tuple[float]
            Effects of mutations in each kind of locus.
        r : int | float | list[float], optional (required if mode='sex' or 'fac_sex')
            Recombination frequency between adjacent loci. If r==0.5 it
            represents a breaking point of a chromosome.
        tau : None | int | list[int], optional (required if mode='fac_sex')
            Number of generations between sexual events.
        sync : None | bool, optional (required if mode='fac_sex')
            Whether facultative sex is synchronous.
        w_only : bool, optional
            Only allow crossovers between fitness loci.  Suppress recombination
            between mutation rate loci.  Loci positions not shuffled.  Default = None.
        u_only : bool, optional
            Only allow crossovers between mutation rate loci.  Suppress recombination
            between fitness loci.  Mutation rate loci positions shuffled; fitness
            loci positions not shuffled.  Default = None.
        nchroms : int, optional
            Number of chromosomes to be generated.  Default = None.

        Returns
        -------
        Haploid

        Examples
        --------
        >>> pop = Haploid.quick_init(mode="sex", ancestor="unmutated", N=10, nloci=(3, 2, 1), base=(.01, .05, .01), s=(.05, .3, .1), r=0.3)
        >>> pop.L
        6
        >>> pop.base_w
        0.01
        >>> pop.s_w
        0.05
        >>> pop.r
        [0.3, 0.3, 0.3, 0.3, 0.3]
        >>> pop.rmap
        [0.3, 0.3, 0.3, 0.3, 0.3]
        """
        gen_map: tuple = Population.get_genetic_map(
            nloci=nloci, r=r, w_only=w_only, u_only=u_only, nchroms=nchroms
        )
        loci_w: list[int] = gen_map[0]
        loci_m: list[int] = gen_map[1]
        loci_a: list[int] = gen_map[2]
        rmap: list[float] = gen_map[3]
        base_w: float = base[0]
        base_m: float = base[1]
        base_a: float = base[2]
        s_w: float = s[0]
        s_m: float = s[1]
        s_a: float = s[2]
        pop = Haploid(
            mode=mode,
            ancestor=ancestor,
            N=N,
            loci_w=loci_w,
            loci_m=loci_m,
            loci_a=loci_a,
            base_w=base_w,
            base_m=base_m,
            base_a=base_a,
            s_w=s_w,
            s_m=s_m,
            s_a=s_a,
            r=rmap,
            tau=tau,
            sync=sync,
        )
        return pop

    def update(self) -> None:
        """Calculate fitness and mutation rate of all individuals."""
        w_mut: npt.NDArray[np.int_] = self.population[:, self.loci_w]
        m_mut: npt.NDArray[np.int_] = self.population[:, self.loci_m]
        a_mut: npt.NDArray[np.int_] = self.population[:, self.loci_a]
        self.w: npt.NDArray[np.float64] = (1 - self.s_w) ** w_mut.sum(axis=1)
        self.mu: npt.NDArray[np.float64] = (1 + self.s_m) ** m_mut.sum(axis=1) * (
            1 - self.s_a
        ) ** a_mut.sum(axis=1)
        self.u_w: npt.NDArray[np.float64] = self.base_w * self.mu
        self.u_m: npt.NDArray[np.float64] = self.base_m * self.mu
        self.u_a: npt.NDArray[np.float64] = self.base_a * self.mu

    def mutate(self) -> None:
        """Mutate loci with probability given by mutation rate."""
        rate: npt.NDArray[np.float64] = np.zeros((self.N, self.L))
        rate[:, self.loci_w] = [[u] for u in self.u_w]
        rate[:, self.loci_m] = [[u] for u in self.u_m]
        rate[:, self.loci_a] = [[u] for u in self.u_a]
        try:
            self.population += rng.binomial(1, rate)
        except ValueError:
            print("*")
            rate[rate > 1.0] = 1.0
            self.population += rng.binomial(1, rate)
        self.population[self.population > 1] = 1
        self.update()

    def asex(self, n: int) -> npt.NDArray[np.int_]:
        """
        Generate n new haploid individuals by asexual reproduction.

        Select haploids and allow them to generate copies of themselves.

        Parameters
        ----------
        n : int
            Number of individuals to generate.

        Returns
        -------
        npt.NDArray[np.int_]
            Population arrays.
        """
        assert n > 0, "Number of individuals to generate must be greater than zero."
        i: npt.NDArray[np.int_] = self.select(n)
        return self.population[i, :]

    def sex(self, n: int) -> npt.NDArray[np.int_]:
        """
        Generate n new haploid individuals by sexual reproduction.

        Select haploids, generate transient diploids from them through fusion, and then
        subject them to meiosis to generate a new set of haploids.

        Parameters
        ----------
        n : int
            Number of individuals to generate.

        Returns
        -------
        npt.NDArray[np.int_]
            Population arrays.
        """
        assert n > 0, "Number of individuals to generate must be greater than zero."
        # natural selection
        i1: npt.NDArray[np.int_] = self.select(n)
        i2: npt.NDArray[np.int_] = self.select(n)
        # fusion
        diploids: npt.NDArray[np.int_] = np.array([self.population[i1, :], self.population[i2, :]])
        # meiosis
        haploids: npt.NDArray[np.int_] = meiosis(diploids, np.arange(n), self.rmap)
        return haploids

    def fac(self) -> npt.NDArray[np.int_]:
        """
        Generate a new haploid population by facultative sexual reproduction.

        Returns
        -------
        npt.NDArray[np.int_]
            Population arrays.
        """
        if isinstance(self.tau, int):
            tau: int = self.tau
        if self.sync:
            if isinstance(self.tau, list):
                if self.nevents < len(self.tau):
                    tau: int = sum(self.tau[: (self.nevents + 1)])
                else:
                    tau = 999999999999999999999
            if (self.t > 0) and (self.t % tau == 0):
                self.nevents += 1
                haploids = self.sex(self.N)
            else:
                haploids = self.asex(self.N)
        else:
            # number of individuals generated sexually
            nsex: int = rng.poisson(self.N / tau)
            if nsex > 0:
                sexuals = self.sex(nsex)
                asexuals = self.asex(self.N - nsex)
                haploids = np.concatenate([sexuals, asexuals])
            else:
                haploids = self.asex(self.N)
        return haploids

    def next_gen(self) -> None:
        """
        Complete one round of the life-cycle including:
        - mutation
        - natural selection
        - reproduction
        """
        self.mutate()
        if self.mode == "asex":
            self.population = self.asex(self.N)
        elif self.mode == "sex":
            self.population = self.sex(self.N)
        elif self.mode == "fac_sex":
            self.population = self.fac()
        self.update()
        self.t += 1

    def reach(self, ptarget: float = 1.0) -> tuple[bool, int]:
        """
        Assay spread of an antimutator mutation in an unmutated population. A
        mutation is introduced at a random antimutator locus and the population
        is allowed to evolve until the mutation either reaches a certain target
        frequency p or disappears. Fitness loci are allowed to mutate but mutator
        and (other) antimutator loci are not.

        Parameters
        ----------
        ptarget : float (default = 1., fixation)
            Target frequency the antimutator mutation must reach.

        Returns
        -------
        tuple[bool, int]
            Whether the mutation reached target frequency (True) or disappeared
            (False) and the time it took to do so.
        """
        assert self.base_m == 0, "Baseline mutation rate of mutator loci must be 0."
        assert self.base_a == 0, "Baseline mutation rate of antimutator loci must be 0."
        assert self.ancestor == "unmutated", "Ancestral population must be unmutated."
        loc = self.loci_a[0]
        self.population[0, loc] = 1
        p = self.population[:, loc].sum() / self.N
        while 0 < p < ptarget:
            self.next_gen()
            p = self.population[:, loc].sum() / self.N
        if p >= ptarget:
            return True, self.t
        else:
            return False, self.t


@dataclass(kw_only=True)
class Diploid(Population):
    """
    Population of diploids with loci controlling fitness and mutation rate.

    Extends Population.

    Additional parameters
    ---------------------
    dom_w : float
        Dominance coefficient of mutations in fitness loci.
    dom_m : float
        Dominance coefficient of mutations in mutator loci.
    dom_a : float
        Dominance coefficient of mutations in antimutator loci.
    """

    modes: ClassVar[frozenset] = frozenset(["asex", "sex", "self", "fac_sex", "fac_self"])
    dom_w: float
    dom_m: float
    dom_a: float

    def __post_init__(self) -> None:
        """Set ploidy.  Validate dominance coefficients."""
        self.ploidy: int = 2
        super().__post_init__()
        # Validate dom_*
        for dom in [self.dom_w, self.dom_m, self.dom_a]:
            assert 0 <= dom <= 1, "Dominance coefficients must be between 0 and 1."

    def _validate_mode(self) -> None:
        """Check validity of reproductive mode."""
        assert self.mode in Diploid.modes, (
            f"Reproductive mode must be one of the following: {', '.join(sorted(Diploid.modes))}."
        )

    def generate_population(self) -> None:
        """
        Generate population.

        Implemented types:
        - unmutated
        - mutated_p where p is the proportion of individuals mutated.
        - all_het (make all loci heterozygous)
        - *_het (make all * loci heterozygous)
        - asex_t where population evolves under obligate asexual reproduction
          for t generations.
        - sex_t where population evolves under obligate sexual reproduction with
          free recombination for t generations.
        """
        unmut: npt.NDArray[np.int_] = np.zeros((2, self.N, self.L), dtype=int)
        self.population: npt.NDArray[np.int_] = unmut
        self.update()
        if self.ancestor == "unmutated":
            pass
        elif self.ancestor[:8] == "mutated_":
            p: float = float(self.ancestor.split("_")[1])
            assert 0 < p < 1, "Proportion of mutations must be between 0 and 1."
            self.population += rng.binomial(1, p, np.shape(unmut))
        elif self.ancestor == "all_het":
            self.population[1] += 1
        elif self.ancestor == "w_het":
            self.population[1, :, self.loci_w] += 1
        elif self.ancestor == "m_het":
            self.population[1, :, self.loci_m] += 1
        elif self.ancestor == "a_het":
            self.population[1, :, self.loci_a] += 1
        elif self.ancestor[:4] == "asex":
            t: int = int(self.ancestor.split("_")[1])
            mode = self.mode
            self.mode = "asex"
            for i in range(t):
                self.next_gen()
            self.mode = mode
        elif self.ancestor[:3] == "sex":
            t: int = int(self.ancestor.split("_")[1])
            mode = self.mode
            rmap = self.rmap
            self.mode = "sex"
            self.rmap = [0.5] * (self.L - 1)
            for i in range(t):
                self.next_gen()
            self.mode = mode
            self.rmap = rmap
        else:
            raise NotImplementedError(f"{self.ancestor} initial state not implemented.")
        self.update()

    @staticmethod
    def quick_init(
        *,
        mode: str,
        ancestor: str,
        N: int,
        nloci: tuple[int, int, int],
        base: tuple[float, float, float],
        s: tuple[float, float, float],
        dom: tuple[float, float, float],
        r: float = 0.0,
        tau: None | int | list[int] = None,
        sync: None | bool = None,
        w_only: None | bool = None,
        u_only: None | bool = None,
        nchroms: None | int = None,
    ) -> Diploid:
        """
        Generate a population of diploid individuals based on a simpler
        parameter specification than that of the __init__() method.

        nloci, base, and s parameters take tuples (w, s, a).

        The loci attributes are generated based on the nloci values.  The loci
        of different types are distributed randomly over a single chromosome.

        Parameters
        ----------
        mode : str
            Mode of reproduction.
        ancestor : str
            Type of ancestral population.
        N : int
            Population size.
        nloci : tuple[int]
            Number of loci of each kind.
        base : tuple[float]
            Baseline mutation rates for each kind of locus.
        s : tuple[float]
            Effects of mutations in each kind of locus.
        dom : tuple[float]
            Dominance coefficients of mutations in each kind of locus.
        r : float
            Recombination frequency between adjacent loci. If r==0.5 it
            represents a breaking point of a chromosome.
        tau : None | int | list[int], optional (required if mode='fac_*')
            Number of generations between sexual events.
        sync : None | bool, optional (required if mode='fac_*')
            Whether facultative sex is synchronous.
        w_only : bool, optional
            Only allow crossovers between fitness loci.  Suppress recombination
            between mutation rate loci.  Loci positions not shuffled.  Default = None.
        u_only : bool, optional
            Only allow crossovers between mutation rate loci.  Suppress recombination
            between fitness loci.  Mutation rate loci positions shuffled; fitness
            loci positions not shuffled.  Default = None.
        nchroms : int, optional
            Number of chromosomes to be generated.  Default = None.

        Returns
        -------
        Diploid

        Examples
        --------
        >>> pop = Diploid.quick_init(mode="fac_self", ancestor="unmutated", N=10, nloci=(3, 2, 1), base=(.01, .05, .01), s=(.05, .3, .1), dom=(.5, .3, .1), r=0.3, tau=10, sync=False)
        >>> pop.L
        6
        >>> pop.base_w
        0.01
        >>> pop.s_w
        0.05
        >>> pop.rmap
        [0.3, 0.3, 0.3, 0.3, 0.3]
        >>> pop.tau
        10
        """
        gen_map: tuple = Population.get_genetic_map(
            nloci=nloci, r=r, w_only=w_only, u_only=u_only, nchroms=nchroms
        )
        loci_w: list[int] = gen_map[0]
        loci_m: list[int] = gen_map[1]
        loci_a: list[int] = gen_map[2]
        rmap: list[float] = gen_map[3]
        base_w: float = base[0]
        base_m: float = base[1]
        base_a: float = base[2]
        s_w: float = s[0]
        s_m: float = s[1]
        s_a: float = s[2]
        dom_w: float = dom[0]
        dom_m: float = dom[1]
        dom_a: float = dom[2]
        pop = Diploid(
            mode=mode,
            ancestor=ancestor,
            N=N,
            loci_w=loci_w,
            loci_m=loci_m,
            loci_a=loci_a,
            base_w=base_w,
            base_m=base_m,
            base_a=base_a,
            s_w=s_w,
            s_m=s_m,
            s_a=s_a,
            dom_w=dom_w,
            dom_m=dom_m,
            dom_a=dom_a,
            r=rmap,
            tau=tau,
            sync=sync,
        )
        return pop

    def update(self) -> None:
        """Calculate fitness and mutation rate of all individuals."""
        w_mut = self.population.sum(axis=0)[:, self.loci_w]
        m_mut = self.population.sum(axis=0)[:, self.loci_m]
        a_mut = self.population.sum(axis=0)[:, self.loci_a]
        w = w_mut + 1.0
        m = m_mut + 1.0
        a = a_mut + 1.0
        w[np.where(w_mut == 1)] = 1 - self.dom_w * self.s_w
        w[np.where(w_mut == 2)] = 1 - self.s_w
        m[np.where(m_mut == 1)] = 1 + self.dom_m * self.s_m
        m[np.where(m_mut == 2)] = 1 + self.s_m
        a[np.where(a_mut == 1)] = 1 - self.dom_m * self.s_a
        a[np.where(a_mut == 2)] = 1 - self.s_a
        self.w = w.prod(axis=1)
        self.mu = m.prod(axis=1) * a.prod(axis=1)
        self.u_w = self.base_w * self.mu
        self.u_m = self.base_m * self.mu
        self.u_a = self.base_a * self.mu

    def mutate(self) -> None:
        """Mutate loci with probability given by mutation rate."""
        rate: npt.NDArray[np.float64] = np.zeros((self.N, self.L))
        rate[:, self.loci_w] = [[u] for u in self.u_w]
        rate[:, self.loci_m] = [[u] for u in self.u_m]
        rate[:, self.loci_a] = [[u] for u in self.u_a]
        rate = np.array([rate, rate])
        try:
            self.population += rng.binomial(1, rate)
        except ValueError:
            print("*")
            rate[rate > 1.0] = 1.0
            self.population += rng.binomial(1, rate)
        self.population[self.population > 1] = 1
        self.update()

    def asex(self, n: int) -> npt.NDArray[np.int_]:
        """
        Generate n new diploid individuals by asexual reproduction.

        Select diploids and allow them to generate copies of themselves.

        Parameters
        ----------
        n : int
            Number of individuals to generate.

        Returns
        -------
        npt.NDArray[np.int_]
            Population arrays.
        """
        assert n > 0, "Number of individuals to generate must be greater than zero."
        i: npt.NDArray[np.int_] = self.select(n)
        return self.population[:, i, :]

    def sex(self, n: int) -> npt.NDArray[np.int_]:
        """
        Generate n new diploid individuals by sexual reproduction.

        Select diploids, generate haploids from them through meiosis, and then
        fuse them to generate a new set of diploids.

        Parameters
        ----------
        n : int
            Number of individuals to generate.

        Returns
        -------
        npt.NDArray[np.int_]
            Population arrays.
        """
        assert n > 0, "Number of individuals to generate must be greater than zero."
        i1: npt.NDArray[np.int_] = self.select(n)
        i2: npt.NDArray[np.int_] = self.select(n)
        haploids1: npt.NDArray[np.int_] = meiosis(self.population, i1, self.rmap)
        haploids2: npt.NDArray[np.int_] = meiosis(self.population, i2, self.rmap)
        diploids: npt.NDArray[np.int_] = np.array([haploids1, haploids2])
        return diploids

    def selfing(self, n: int) -> npt.NDArray[np.int_]:
        """
        Generate n new diploid individuals by selfing.

        Parameters
        ----------
        n : int
            Number of individuals to generate.

        Returns
        -------
        npt.NDArray[np.int_]
            Population arrays.
        """
        assert n > 0, "Number of individuals to generate must be greater than zero."
        i: npt.NDArray[np.int_] = self.select(n)
        haploids1: npt.NDArray[np.int_] = meiosis(self.population, i, self.rmap)
        haploids2: npt.NDArray[np.int_] = meiosis(self.population, i, self.rmap)
        diploids: npt.NDArray[np.int_] = np.array([haploids1, haploids2])
        return diploids

    def fac(self) -> npt.NDArray[np.int_]:
        """
        Generate a new haploid population by facultative sexual reproduction (outcrossing) or selfing.

        Returns
        -------
        np.array[int]
            Population arrays.
        """
        if isinstance(self.tau, int):
            tau: int = self.tau
        if self.sync:
            if isinstance(self.tau, list):
                if self.nevents < len(self.tau):
                    tau: int = sum(self.tau[: (self.nevents + 1)])
                else:
                    tau = 999999999999999999999
            if (self.t > 0) and (self.t % tau == 0):
                self.nevents += 1
                if self.mode == "fac_sex":
                    diploids: npt.NDArray[np.int_] = self.sex(self.N)
                elif self.mode == "fac_self":
                    diploids = self.selfing(self.N)
            else:
                diploids = self.asex(self.N)
        else:
            # number of individuals generated sexually
            nsex: int = rng.poisson(self.N / tau)
            if nsex > 0:
                if self.mode == "fac_sex":
                    sexuals: npt.NDArray[np.int_] = self.sex(nsex)
                elif self.mode == "fac_self":
                    sexuals = self.selfing(nsex)
                asexuals: npt.NDArray[np.int_] = self.asex(self.N - nsex)
                diploids = np.array(
                    [
                        np.concatenate([sexuals[0], asexuals[0]]),
                        np.concatenate([sexuals[1], asexuals[1]]),
                    ]
                )
            else:
                diploids = self.asex(self.N)
        return diploids

    def next_gen(self) -> None:
        """
        Complete one round of the life-cycle including:
        - mutation
        - natural selection
        - reproduction
        """
        self.mutate()
        if self.mode == "asex":
            self.population = self.asex(self.N)
        elif self.mode == "sex":
            self.population = self.sex(self.N)
        elif self.mode == "self":
            self.population = self.selfing(self.N)
        elif (self.mode == "fac_sex") or (self.mode == "fac_self"):
            self.population = self.fac()
        self.update()
        self.t += 1

    def reach(self, ptarget: float = 1.0) -> tuple[bool, int]:
        """
        Assay spread of an antimutator mutation in an unmutated population. A
        mutation is introduced at a random antimutator locus in a heterozygous
        state and the population is allowed to evolve until the mutation either
        reaches a certain target frequency p or disappears. Fitness loci are
        allowed to mutate but mutator and (other) antimutator loci are not.

        In obligate asexuals, p <= 0.5 because the antimutator mutation can
        never occur in a homozygous state.

        Parameters
        ----------
        ptarget : float (default = 1., fixation)
            Target frequency the antimutator mutation must reach.

        Returns
        -------
        tuple[bool, int]
            Whether the mutation reached target frequency (True) or disappeared
            (False) and the time it took to do so.
        """
        assert self.base_m == 0, "Baseline mutation rate of mutator loci must be 0."
        assert self.base_a == 0, "Baseline mutation rate of antimutator loci must be 0."
        assert self.ancestor == "unmutated", "Ancestral population must be unmutated."
        loc = self.loci_a[0]
        self.population[0, 0, loc] = 1
        if self.mode == "asex":
            assert ptarget <= 0.5, "In an asexual diploid, the allele frequency cannot exceed 0.5."
        p = self.population[:, :, loc].sum() / (2 * self.N)
        while 0 < p < ptarget:
            self.next_gen()
            p = self.population[:, :, loc].sum() / (2 * self.N)
        if p >= ptarget:
            return True, self.t
        else:
            return False, self.t


@dataclass(kw_only=True)
class Triploid(Population):
    """
    Population of triploids with loci controlling fitness and mutation rate.

    Extends Population.

    Additional parameters
    ---------------------
    dom_w : float
        Dominance coefficient of mutations in fitness loci.
    dom_m : float
        Dominance coefficient of mutations in mutator loci.
    dom_a : float
        Dominance coefficient of mutations in antimutator loci.
    """

    modes: ClassVar[frozenset] = frozenset(["asex"])
    dom_w: float
    dom_m: float
    dom_a: float

    def __post_init__(self) -> None:
        """Set ploidy.  Validate dominance coefficients."""
        self.ploidy: int = 3
        super().__post_init__()
        # Validate dom_*
        for dom in [self.dom_w, self.dom_m, self.dom_a]:
            assert 0 <= dom <= 1, "Dominance coefficients must be between 0 and 1."

    def _validate_mode(self) -> None:
        """Check validity of reproductive mode."""
        assert self.mode in Triploid.modes, "Reproductive mode must be asex."

    def generate_population(self) -> None:
        """
        Generate population.

        Implemented types:
        - unmutated
        - mutated_p where p is the proportion of individuals mutated.
        """
        unmut: npt.NDArray[np.int_] = np.zeros((3, self.N, self.L), dtype=int)
        self.population: npt.NDArray[np.int_] = unmut
        self.update()
        if self.ancestor == "unmutated":
            pass
        elif self.ancestor[:8] == "mutated_":
            p: float = float(self.ancestor.split("_")[1])
            assert 0 < p < 1, "Proportion of mutations must be between 0 and 1."
            self.population += rng.binomial(1, p, np.shape(unmut))
        else:
            raise NotImplementedError(f"{self.ancestor} initial state not implemented.")
        self.update()

    @staticmethod
    def quick_init(
        *,
        mode: str,
        ancestor: str,
        N: int,
        nloci: tuple[int, int, int],
        base: tuple[float, float, float],
        s: tuple[float, float, float],
        dom: tuple[float, float, float],
    ) -> Triploid:
        """
        Generate a population of diploid individuals based on a simpler
        parameter specification than that of the __init__() method.

        nloci, base, and s parameters take tuples (w, s, a).

        The loci attributes are generated based on the nloci values.  The loci
        of different types are distributed randomly over a single chromosome.

        Parameters
        ----------
        mode : str
            Mode of reproduction.
        ancestor : str
            Type of ancestral population.
        N : int
            Population size.
        nloci : tuple[int]
            Number of loci of each kind.
        base : tuple[float]
            Baseline mutation rates for each kind of locus.
        s : tuple[float]
            Effects of mutations in each kind of locus.
        dom : tuple[float]
            Dominance coefficients of mutations in each kind of locus.

        Returns
        -------
        Triploid

        Examples
        --------
        >>> pop = Triploid.quick_init(mode="asex", ancestor="unmutated", N=10, nloci=(3, 2, 1), base=(.01, .05, .01), s=(.05, .3, .1), dom=(.5, .3, .1))
        >>> pop.L
        6
        >>> pop.base_w
        0.01
        >>> pop.s_w
        0.05
        """
        gen_map: tuple = Population.get_genetic_map(nloci=nloci, r=0)
        loci_w: list[int] = gen_map[0]
        loci_m: list[int] = gen_map[1]
        loci_a: list[int] = gen_map[2]
        base_w: float = base[0]
        base_m: float = base[1]
        base_a: float = base[2]
        s_w: float = s[0]
        s_m: float = s[1]
        s_a: float = s[2]
        dom_w: float = dom[0]
        dom_m: float = dom[1]
        dom_a: float = dom[2]
        pop = Triploid(
            mode=mode,
            ancestor=ancestor,
            N=N,
            loci_w=loci_w,
            loci_m=loci_m,
            loci_a=loci_a,
            base_w=base_w,
            base_m=base_m,
            base_a=base_a,
            s_w=s_w,
            s_m=s_m,
            s_a=s_a,
            dom_w=dom_w,
            dom_m=dom_m,
            dom_a=dom_a,
        )
        return pop

    def update(self) -> None:
        """Calculate fitness and mutation rate of all individuals."""
        w_mut = self.population.sum(axis=0)[:, self.loci_w]
        m_mut = self.population.sum(axis=0)[:, self.loci_m]
        a_mut = self.population.sum(axis=0)[:, self.loci_a]
        w = w_mut + 1.0
        m = m_mut + 1.0
        a = a_mut + 1.0
        # placeholder -- figure out corrections for dominance coefficients for
        # one and two copies of an allele
        h1 = 1
        h2 = 1
        w[np.where(w_mut == 1)] = 1 - h1 * self.dom_w * self.s_w
        w[np.where(w_mut == 2)] = 1 - h2 * self.dom_w * self.s_w
        w[np.where(w_mut == 3)] = 1 - self.s_w
        m[np.where(m_mut == 1)] = 1 + h1 * self.dom_m * self.s_m
        m[np.where(m_mut == 2)] = 1 + h2 * self.dom_m * self.s_m
        m[np.where(m_mut == 3)] = 1 + self.s_m
        a[np.where(a_mut == 1)] = 1 - h1 * self.dom_m * self.s_a
        a[np.where(a_mut == 2)] = 1 - h2 * self.dom_m * self.s_a
        a[np.where(a_mut == 3)] = 1 - self.s_a
        self.w = w.prod(axis=1)
        self.mu = m.prod(axis=1) * a.prod(axis=1)
        self.u_w = self.base_w * self.mu
        self.u_m = self.base_m * self.mu
        self.u_a = self.base_a * self.mu

    def mutate(self) -> None:
        """Mutate loci with probability given by mutation rate."""
        rate: npt.NDArray[np.float64] = np.zeros((self.N, self.L))
        rate[:, self.loci_w] = [[u] for u in self.u_w]
        rate[:, self.loci_m] = [[u] for u in self.u_m]
        rate[:, self.loci_a] = [[u] for u in self.u_a]
        rate = np.array([rate, rate, rate])
        try:
            self.population += rng.binomial(1, rate)
        except ValueError:
            print("*")
            rate[rate > 1.0] = 1.0
            self.population += rng.binomial(1, rate)
        self.population[self.population > 1] = 1
        self.update()

    def asex(self, n: int) -> npt.NDArray[np.int_]:
        """
        Generate n new triploid individuals by asexual reproduction.

        Select triploids and allow them to generate copies of themselves.

        Parameters
        ----------
        n : int
            Number of individuals to generate.

        Returns
        -------
        npt.NDArray[np.int_]
            Population arrays.
        """
        assert n > 0, "Number of individuals to generate must be greater than zero."
        i: npt.NDArray[np.int_] = self.select(n)
        return self.population[:, i, :]

    def sex(self, n: int) -> None:
        """Generate individuals by sexual reproduction. Not implemented because disallowed."""

    def fac(self) -> None:
        """Generate a new population by facultative sexual reproduction. Not implemented because disallowed."""

    def next_gen(self) -> None:
        """
        Complete one round of the life-cycle including:
        - mutation
        - natural selection
        - reproduction
        """
        self.mutate()
        self.population = self.asex(self.N)
        self.update()
        self.t += 1

    def reach(self, ptarget: float = 1 / 3) -> tuple[bool, int]:
        """
        Assay spread of an antimutator mutation in an unmutated population. A
        mutation is introduced at a random antimutator locus in a heterozygous
        state and the population is allowed to evolve until the mutation either
        reaches a certain target frequency p or disappears. Fitness loci are
        allowed to mutate but mutator and (other) antimutator loci are not.

        Because triploids are obligate asexuals, p <= 1/3 because the
        antimutator mutation can never occur in a homozygous state.

        Parameters
        ----------
        ptarget : float (default = 1/3, fixation)
            Target frequency the antimutator mutation must reach.

        Returns
        -------
        tuple[bool, int]
            Whether the mutation reached target frequency (True) or disappeared
            (False) and the time it took to do so.
        """
        assert self.base_m == 0, "Baseline mutation rate of mutator loci must be 0."
        assert self.base_a == 0, "Baseline mutation rate of antimutator loci must be 0."
        assert self.ancestor == "unmutated", "Ancestral population must be unmutated."
        loc = self.loci_a[0]
        self.population[0, 0, loc] = 1

        assert ptarget <= 1 / 3, "In an asexual triploid, the allele frequency cannot exceed 1/3."
        p = self.population[:, :, loc].sum() / (3 * self.N)
        while 0 < p < ptarget:
            self.next_gen()
            p = self.population[:, :, loc].sum() / (3 * self.N)
        if p >= ptarget:
            return True, self.t
        else:
            return False, self.t


# =============================================================================
# Parallel simulation
# =============================================================================


def init_rng():
    """Initialize random number generator."""
    global rng
    rng = rnd.default_rng()
    print(rng.bit_generator.state["state"])


def multi_evolve(
    *,
    ploidy: int,
    mode: str,
    ancestor: str,
    N: int,
    nloci: tuple[int, int, int],
    base: tuple[float, float, float],
    s: tuple[float, float, float],
    r: float,
    ngens: int,
    every: int,
    nreps: int,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a population and run a simulation.

    Use in multiprocessor simulations.

    Parameters
    ----------
    ploidy : int
        Ploidy (1 or 2).
    mode : str
        Mode of reproduction.
    N : int
        Population size.
    nloci : tuple[int]
        Number of loci of each kind.
    base : tuple[float]
        Baseline mutation rates for each kind of locus.
    s : tuple[float]
        Effects of mutations in each kind of locus.
    ngens : int
        Number of generations.
    every : int
        Number of generations between data collection events.
        (1 means that data are collected every generation).
    nreps : int
        Number of replicate populations.
    r : float
        Recombination frequency between adjacent loci.
    tau : None | int | list[int], optional (required if mode='fac_*')
        Number of generations between sexual events.
    sync : None | bool, optional (required if mode='fac_*')
        Whether facultative sex is synchronous.

    Returns
    -------
    tuple[pd.DataFrame]
        Time series for each replicate in columns. Separate data frames for:
        - Mean fitness
        - Mean deleterious mutation rate
        - Variance in fitness
        - Variance in deleterious mutation rate
        - Covariance between deleterious mutation rate and fitness
    """
    if ploidy == 1:
        del kwargs["dom"]
        hpop: Haploid = Haploid.quick_init(
            mode=mode, ancestor=ancestor, N=N, nloci=nloci, base=base, s=s, r=r, **kwargs
        )
        return hpop.replicate_evolve(ngens, every, nreps)
    elif ploidy == 2:
        dpop: Diploid = Diploid.quick_init(
            mode=mode, ancestor=ancestor, N=N, nloci=nloci, base=base, s=s, r=r, **kwargs
        )
        return dpop.replicate_evolve(ngens, every, nreps)
    else:
        raise ValueError("Ploidy must be 1 or 2.")


def multi_reach(
    *,
    ploidy: int,
    mode: str,
    N: int,
    nloci: tuple[int, int, int],
    base: tuple[float, float, float],
    s: tuple[float, float, float],
    r: float,
    ptarget: float,
    nreps: int,
    **kwargs,
) -> tuple[int, list[int]]:
    """
    Create a population and run a 'reach' simulation.

    Use in multiprocessor simulations.

    Parameters
    ----------
    ploidy : int
        Ploidy (1 or 2).
    mode : str
        Mode of reproduction.
    N : int
        Population size.
    nloci : tuple[int]
        Number of loci of each kind.
    base : tuple[float]
        Baseline mutation rates for each kind of locus.
    s : tuple[float]
        Effects of mutations in each kind of locus.
    r : float
        Recombination frequency between adjacent loci.
    nreps : int
        Number of replicate populations.
    ptarget : float (default = 1., fixation)
        Target frequency the antimutator mutation must reach.
    tau : None | int | list[int], optional (required if mode='fac_*')
        Number of generations between sexual events.
    sync : None | bool, optional (required if mode='fac_*')
        Whether facultative sex is synchronous.

    Returns
    -------
    tuple[int, list[int]]
        Number of times a population reached the target frequency and list
        containing times.
    """
    if ploidy == 1:
        del kwargs["dom"]
        hpop: Haploid = Haploid.quick_init(
            mode=mode, ancestor="unmutated", N=N, nloci=nloci, base=base, s=s, r=r, **kwargs
        )
        return hpop.replicate_reach(ptarget, nreps)
    elif ploidy == 2:
        dpop: Diploid = Diploid.quick_init(
            mode=mode, ancestor="unmutated", N=N, nloci=nloci, base=base, s=s, r=r, **kwargs
        )
        return dpop.replicate_reach(ptarget, nreps)
    else:
        raise ValueError("Ploidy must be 1 or 2.")


def struple(x: tuple) -> str:
    """
    Create string from tuple with elements separated by a space.

    Parameters
    ----------
    x : tuple
        Tuple.

    Returns
    -------
    str
        String with elements of tuple.
    """
    xstr: str = ""
    for i in x:
        xstr += f"{i} "
    return xstr[:-1]


# =============================================================================
# Main functions
# =============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments to module.

    Returns
    -------
    argparse.Namespace
        Command-line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    # Modes: --test, --sim, --multi
    parser.add_argument("--test", action="store_true", help="Run doctests.")
    parser.add_argument("--sim", action="store_true", help="Run simulation.")
    parser.add_argument("--multi", action="store_true", help="Run multiprocessing simulation.")
    # Parameters
    # --sim and --multi
    parser.add_argument("--ploidy", type=int, help="Ploidy. (Required for --sim and --multi.)")
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode of reproduction. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "--ancestor",
        type=str,
        help="Type of ancestral population. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "-N", "--N", type=int, help="Population size. (Required for --sim and --multi.)"
    )
    parser.add_argument(
        "-L",
        "--nloci",
        type=int,
        nargs="+",
        help="Numbers of fitness, mutator, and antimutator loci. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=float,
        nargs="+",
        help="Baseline mutation rates of fitness, mutator, and antimutator loci. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "-s",
        "--s",
        type=float,
        nargs="+",
        help="Effects of mutations on fitness, mutator, and antimutator loci. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "-d",
        "--dom",
        type=float,
        nargs="+",
        help="Dominance coefficients of mutations on fitness, mutator, and antimutator loci. (Required for --sim and --multi with diploids.)",
    )
    parser.add_argument(
        "-r",
        "--r",
        type=float,
        help="Recombination frequency. (Required for --sim and --multi when mode is sex or fac.)",
    )
    parser.add_argument(
        "-t",
        "--tau",
        type=int,
        help="Number of generations between sexual events. (Required for --sim and --multi when mode is fac.)",
    )
    parser.add_argument("--sync", action="store_true", help="Synchronous facultative sex.")
    parser.add_argument(
        "--w_only",
        action="store_true",
        help="Recombination only in mutation rate loci.",
    )
    parser.add_argument("--u_only", action="store_true", help="Recombination only in fitness loci.")
    parser.add_argument("--nchroms", type=int, help="Number of chromosomes.")
    parser.add_argument(
        "--ngens",
        type=int,
        help="Number of generations. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "--every",
        type=int,
        help="Number of generations between data collection events. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "--nreps",
        type=int,
        help="Number of replicates. (Required for --sim and --multi.)",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="Output file name prefix. (Required for --sim and --multi.)",
    )
    # --multi only
    parser.add_argument(
        "--nproc", type=int, help="Number of processors to use. (Required for --multi.)"
    )
    # reach a target frequency
    parser.add_argument(
        "--reach",
        action="store_true",
        help="Whether to assay antimutator allele reaching a target frequency.",
    )
    # target frequency
    parser.add_argument(
        "--ptarget",
        type=float,
        help="Frequency antimutator allele must reach. (Required for --reach.)",
    )
    args = parser.parse_args()
    if args.sim or args.multi:
        args.nloci = tuple(args.nloci)
        args.base = tuple(args.base)
        if args.ploidy == 2:
            args.dom = tuple(args.dom)
        args.s = tuple(args.s)
        if args.mode[:3] != "fac":
            args.sync = None
            args.tau = None
    return args


def timer(fun):
    @wraps(fun)
    def wrapper(args: argparse.Namespace) -> float:
        t0: float = time.time()
        fun(args)
        t1: float = time.time()
        deltat: float = t1 - t0
        time.sleep(3)  # ensure that the following is printed at the bottom
        if deltat < 100:
            print(f"\nElapsed time: {round(deltat, 2)} seconds.")
        elif deltat < 6000:
            print(f"\nElapsed time: {round(deltat / 60, 2)} minutes.")
        else:
            print(f"\nElapsed time: {round(deltat / 3600, 2)} hours")
        return deltat

    return wrapper


def main_test() -> None:
    """
    Run doctests.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Python: {sys.version}")
    print("\nPackages used:")
    for i in [np, pd]:
        print(f"  {str(i).split(' ')[1]} {i.__version__}")
    print("\nTesting Started!\n")
    test = doctest.testmod()
    print(f"   {test[1]} tests conducted")
    print(f"   {test[0]} tests failed")
    print("\nTesting Completed!")


def main_info(args: argparse.Namespace) -> None:
    """
    Display parameter values of simulation.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print("--------------------------------------------")
    print("Simulation of the evolution of mutation rate")
    print("--------------------------------------------")
    print("\nDate/time:", datetime.now().ctime())
    print("\n Filename:", args.filename)
    print("\nParameters\n")
    print("   ploidy:", args.ploidy)
    print("     mode:", args.mode)
    print(" ancestor:", args.ancestor)
    print("        N:", args.N)
    print("    nloci:", args.nloci)
    print("     base:", args.base)
    print("        s:", args.s)
    if args.ploidy == 2:
        print("      dom:", args.dom)
    if args.reach:
        print(f"    ngens: time for antimutator mutation to reach p={args.ptarget}")
    else:
        print("    ngens:", args.ngens)
        print(f"    every: {args.every} generations")
    print("        r:", args.r)
    if args.w_only:
        print("           -> Crossovers in fitness loci only.\n")
    elif args.u_only:
        print("           -> Crossovers in mutation rate loci only.\n")
    elif args.nchroms:
        print("  nchroms:", args.nchroms)
    if args.mode[:3] == "fac":
        print(f"      tau: {args.tau} generations")
        print("     sync:", args.sync)
    print("    nreps:", args.nreps)
    if args.multi:
        print("    nproc:", args.nproc)
    else:
        print("    nproc: 1")
        print("\nRandom number generator: ", rng.bit_generator.state)


@timer
def main_sim(args: argparse.Namespace) -> None:
    """
    Run single processor simulation from the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    main_info(args)
    print("Simulation Started!")
    if args.ploidy == 1:
        hpop: Haploid = Haploid.quick_init(
            mode=args.mode,
            ancestor="unmutated",
            N=args.N,
            nloci=args.nloci,
            base=args.base,
            s=args.s,
            r=args.r,
            tau=args.tau,
            sync=args.sync,
            w_only=args.w_only,
            u_only=args.u_only,
            nchroms=args.nchroms,
        )
        if args.reach:
            nreach, treach = hpop.replicate_reach(args.ptarget, args.nreps)
            print(f"nreach = {nreach}")
            print(f"treach = {treach}")
        else:
            hpop.replicate_evolve(args.ngens, args.every, args.nreps, args.filename)
    elif args.ploidy == 2:
        dpop: Diploid = Diploid.quick_init(
            mode=args.mode,
            ancestor="unmutated",
            N=args.N,
            nloci=args.nloci,
            base=args.base,
            s=args.s,
            dom=args.dom,
            r=args.r,
            tau=args.tau,
            sync=args.sync,
            w_only=args.w_only,
            u_only=args.u_only,
            nchroms=args.nchroms,
        )
        if args.reach:
            nreach, treach = dpop.replicate_reach(args.ptarget, args.nreps)
            print(f"nreach = {nreach}")
            print(f"treach = {treach}")
        else:
            dpop.replicate_evolve(args.ngens, args.every, args.nreps, args.filename)
    else:
        raise ValueError("Ploidy must be either 1 or 2.")
    print("Simulation Completed!")


@timer
def main_multi(args: argparse.Namespace) -> None:
    """
    Run multiprocessor simulation from the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    main_info(args)
    print("\nJob Started!")
    print("\nRandom number generators:\n")
    assert args.nreps % args.nproc == 0, (
        "Number of replicates (nreps) must be a multiple of number of processors (nproc)."
    )
    pool: mp.pool.Pool = mp.Pool(processes=args.nproc, initializer=init_rng)  # type: ignore
    subreps: int = args.nreps // args.nproc
    if args.reach:
        sims = [
            pool.apply_async(
                multi_reach,
                (),
                {
                    "ploidy": args.ploidy,
                    "mode": args.mode,
                    "N": args.N,
                    "nloci": args.nloci,
                    "base": args.base,
                    "s": args.s,
                    "nreps": subreps,
                    "dom": args.dom,
                    "r": args.r,
                    "ptarget": args.ptarget,
                    "tau": args.tau,
                    "sync": args.sync,
                    "w_only": args.w_only,
                    "u_only": args.u_only,
                    "nchroms": args.nchroms,
                },
            )
            for i in range(args.nproc)
        ]
        output = [sim.get() for sim in sims]
        nreach = 0
        treach = []
        for out in output:
            nreach += out[0]
            treach += out[1]
        treach = np.array(treach)
        preach = nreach / args.nreps
        print("\nResults:")
        print(f"  mode={args.mode}")
        if args.mode == "sex":
            print(f"     r={args.r}")
        elif args.mode[:3] == "fac":
            print(f"     r={args.r}")
            print(f"   tau={args.tau}")
            print(f"  sync={args.sync}")
        print(f"nreach={nreach} (n={args.nreps})")
        print(f"preach={preach:.6f} (SE={np.sqrt(preach * (1 - preach) / args.nreps):.6f})")
        print(f"treach={treach.mean():.2f} (SD={treach.std(ddof=1):.2f}, max={treach.max()})")
    else:
        sims = [
            pool.apply_async(
                multi_evolve,
                (),
                {
                    "ploidy": args.ploidy,
                    "mode": args.mode,
                    "ancestor": args.ancestor,
                    "N": args.N,
                    "nloci": args.nloci,
                    "base": args.base,
                    "s": args.s,
                    "ngens": args.ngens,
                    "every": args.every,
                    "nreps": subreps,
                    "dom": args.dom,
                    "r": args.r,
                    "tau": args.tau,
                    "sync": args.sync,
                    "w_only": args.w_only,
                    "u_only": args.u_only,
                    "nchroms": args.nchroms,
                },
            )
            for i in range(args.nproc)
        ]
        output = [sim.get() for sim in sims]
        wdata = pd.concat([out[0] for out in output], axis=1)
        udata = pd.concat([out[1] for out in output], axis=1)
        vwdata = pd.concat([out[2] for out in output], axis=1)
        vudata = pd.concat([out[3] for out in output], axis=1)
        covdata = pd.concat([out[4] for out in output], axis=1)
        for data in [wdata, udata, vwdata, vudata, covdata]:
            data.columns = [i for i in range(args.nreps)]
        wdata.to_csv(args.filename + "_w.csv", float_format="%.6g")
        udata.to_csv(args.filename + "_u.csv", float_format="%.6g")
        vwdata.to_csv(args.filename + "_vw.csv", float_format="%.6g")
        vudata.to_csv(args.filename + "_vu.csv", float_format="%.6g")
        covdata.to_csv(args.filename + "_cov.csv", float_format="%.6g")
    print("Job Completed!")


if __name__ == "__main__":
    __spec__ = None
    args: argparse.Namespace = parse_args()
    assert not (args.sim and args.multi), "Select either --sim or --multi."
    if args.test:
        main_test()
    if args.sim:
        main_sim(args)
    elif args.multi:
        main_multi(args)
