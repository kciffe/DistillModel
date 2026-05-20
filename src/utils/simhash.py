import hashlib
from collections import defaultdict
from collections.abc import Iterable


DEFAULT_SIMHASH_BITS = 64


def simhash(text: str, bits: int = DEFAULT_SIMHASH_BITS) -> int:
    weights = [0] * bits
    tokens = [text[i : i + 2] for i in range(max(len(text) - 1, 1))]

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        for bit_index in range(bits):
            if value & (1 << bit_index):
                weights[bit_index] += 1
            else:
                weights[bit_index] -= 1

    fingerprint = 0
    for bit_index, weight in enumerate(weights):
        if weight > 0:
            fingerprint |= 1 << bit_index
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class SimHashBucketIndex:
    def __init__(
        self,
        *,
        threshold: int,
        bits: int = DEFAULT_SIMHASH_BITS,
        bucket_count: int = 4,
    ) -> None:
        self.threshold = threshold
        self.bits = bits
        self.bucket_count = bucket_count
        self.bucket_size = bits // bucket_count
        self._buckets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
        self._texts: set[str] = set()

    def _bucket_keys(self, fingerprint: int) -> Iterable[tuple[int, int]]:
        mask = (1 << self.bucket_size) - 1
        for bucket_index in range(self.bucket_count):
            yield bucket_index, (fingerprint >> (bucket_index * self.bucket_size)) & mask

    def add(self, text: str, fingerprint: int | None = None) -> int:
        normalized_text = text.strip()
        fingerprint = simhash(normalized_text, self.bits) if fingerprint is None else fingerprint
        self._texts.add(normalized_text)
        for bucket_key in self._bucket_keys(fingerprint):
            self._buckets[bucket_key].append((fingerprint, normalized_text))
        return fingerprint

    def find_duplicate(self, text: str, fingerprint: int | None = None) -> str | None:
        normalized_text = text.strip()
        if normalized_text in self._texts:
            return normalized_text

        fingerprint = simhash(normalized_text, self.bits) if fingerprint is None else fingerprint
        checked: set[tuple[int, str]] = set()
        for bucket_key in self._bucket_keys(fingerprint):
            for candidate_fingerprint, candidate_text in self._buckets.get(bucket_key, []):
                candidate = (candidate_fingerprint, candidate_text)
                if candidate in checked:
                    continue
                checked.add(candidate)
                if hamming_distance(fingerprint, candidate_fingerprint) <= self.threshold:
                    return candidate_text
        return None
