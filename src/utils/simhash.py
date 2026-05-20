import hashlib


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
