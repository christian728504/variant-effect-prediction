"""Reference-genome access for variant scoring.

RefGenome wraps pysam.FastaFile with a small extract_alleles helper that returns
the ref-allele and alt-allele sequences of a fixed context length, centered on
the variant position. Indels are length-normalized by trimming the downstream flank.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pysam


_VALID = frozenset("ACGTN")


@dataclass
class RefGenome:
    fasta_path: str | Path

    @cached_property
    def _fa(self) -> pysam.FastaFile:
        return pysam.FastaFile(str(self.fasta_path))

    def length(self, chrom: str) -> int:
        return self._fa.get_reference_length(chrom)

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """0-based half-open fetch, uppercased."""
        return self._fa.fetch(chrom, start, end).upper()

    def extract_alleles(
        self,
        chrom: str,
        start: int,
        allele1: str,
        allele2: str,
        context_len: int,
    ) -> tuple[str, str]:
        """Return (allele1_seq, allele2_seq), both length-`context_len` uppercase.

        Follows the ChromBPNet variant-scorer convention (`variant_generator.py`,
        `__get_allele_seq__`): fetch the genomic flank centered on `start` and
        substitute each allele at the center — there is **no** check that the
        genome base equals allele1, so the reference may carry either allele. The
        downstream flank is truncated so both sequences are exactly `context_len`
        long (length-normalizing any indel). `start` is the 0-based variant
        position where the first allele base is placed.

        Variants near a chromosome boundary are **N-padded** (not dropped or
        errored): missing upstream bases become leading ``N``s and missing
        downstream bases become trailing ``N``s, so the window is always exactly
        `context_len` long. (variant-scorer instead filters such variants out;
        we pad so callers never lose or crash on boundary-proximal variants.)
        """
        allele1 = allele1.upper()
        allele2 = allele2.upper()
        if not (set(allele1) <= _VALID and set(allele2) <= _VALID):
            raise ValueError(
                f"allele1/allele2 contain non-ACGTN bases: "
                f"allele1={allele1!r}, allele2={allele2!r}"
            )

        chrom_len = self.length(chrom)
        center = context_len // 2
        a1_len = len(allele1)

        # Left flank: `center` genomic bases before the variant, N-padded if the
        # window runs off the chromosome start.
        genome_left_start = max(0, start - center)
        left = (
            self.fetch(chrom, genome_left_start, start)
            if start > genome_left_start
            else ""
        )
        left = "N" * (center - len(left)) + left

        # Right flank: genomic bases after the reference-allele region. Fetch a
        # generous span; right-padding below tops it up if the chromosome ends.
        right_start = start + a1_len
        right_end = min(right_start + context_len, chrom_len)
        right = (
            self.fetch(chrom, right_start, right_end) if right_end > right_start else ""
        )

        # Splice allele in at the center; right-pad with N then trim to context_len.
        allele1_seq = (left + allele1 + right + "N" * context_len)[:context_len]
        allele2_seq = (left + allele2 + right + "N" * context_len)[:context_len]
        return allele1_seq, allele2_seq
