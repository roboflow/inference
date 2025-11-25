"""
Mask decoding utilities for the dataset.

The mask in the dataset undergoes a two-step encoding process:
1. First encoded in the COCO format (Run Length Encoding)
2. Then encoded as LZ string for compression

This module reverses that process:
1. First decode from LZ string format (based on https://github.com/marcel-dancak/lz-string-python/blob/master/lzstring.py)
2. Then use pycocotools to decode the COCO RLE format into a mask array
"""

import math
from pycocotools import mask as mask_utils


class Object:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


KEYSTRURISAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-$"
base_reverse_dict = {}


def get_base_value(alphabet, character):
    if alphabet not in base_reverse_dict:
        base_reverse_dict[alphabet] = {}
    for i in range(len(alphabet)):
        base_reverse_dict[alphabet][alphabet[i]] = i
    return base_reverse_dict[alphabet][character]


def decompress(length, reset_value, get_next_value):
    dictionary = {}
    enlarge_in = 4
    dict_size = 4
    num_bits = 3
    entry = ""
    result = []

    data = Object(val=get_next_value(0), position=reset_value, index=1)

    for i in range(3):
        dictionary[i] = i

    bits = 0
    maxpower = math.pow(2, 2)
    power = 1

    while power != maxpower:
        resb = data.val & data.position
        data.position >>= 1
        if data.position == 0:
            data.position = reset_value
            data.val = get_next_value(data.index)
            data.index += 1

        bits |= power if resb > 0 else 0
        power <<= 1

    next = bits
    if next == 0:
        bits = 0
        maxpower = math.pow(2, 8)
        power = 1
        while power != maxpower:
            resb = data.val & data.position
            data.position >>= 1
            if data.position == 0:
                data.position = reset_value
                data.val = get_next_value(data.index)
                data.index += 1
            bits |= power if resb > 0 else 0
            power <<= 1
        c = chr(bits)
    elif next == 1:
        bits = 0
        maxpower = math.pow(2, 16)
        power = 1
        while power != maxpower:
            resb = data.val & data.position
            data.position >>= 1
            if data.position == 0:
                data.position = reset_value
                data.val = get_next_value(data.index)
                data.index += 1
            bits |= power if resb > 0 else 0
            power <<= 1
        c = chr(bits)
    elif next == 2:
        return ""

    dictionary[3] = c
    w = c
    result.append(c)
    counter = 0
    while True:
        counter += 1
        if data.index > length:
            return ""

        bits = 0
        maxpower = math.pow(2, num_bits)
        power = 1
        while power != maxpower:
            resb = data.val & data.position
            data.position >>= 1
            if data.position == 0:
                data.position = reset_value
                data.val = get_next_value(data.index)
                data.index += 1
            bits |= power if resb > 0 else 0
            power <<= 1

        c = bits
        if c == 0:
            bits = 0
            maxpower = math.pow(2, 8)
            power = 1
            while power != maxpower:
                resb = data.val & data.position
                data.position >>= 1
                if data.position == 0:
                    data.position = reset_value
                    data.val = get_next_value(data.index)
                    data.index += 1
                bits |= power if resb > 0 else 0
                power <<= 1

            dictionary[dict_size] = chr(bits)
            dict_size += 1
            c = dict_size - 1
            enlarge_in -= 1
        elif c == 1:
            bits = 0
            maxpower = math.pow(2, 16)
            power = 1
            while power != maxpower:
                resb = data.val & data.position
                data.position >>= 1
                if data.position == 0:
                    data.position = reset_value
                    data.val = get_next_value(data.index)
                    data.index += 1
                bits |= power if resb > 0 else 0
                power <<= 1
            dictionary[dict_size] = chr(bits)
            dict_size += 1
            c = dict_size - 1
            enlarge_in -= 1
        elif c == 2:
            return "".join(result)

        if enlarge_in == 0:
            enlarge_in = math.pow(2, num_bits)
            num_bits += 1

        if c in dictionary:
            entry = dictionary[c]
        else:
            if c == dict_size:
                entry = w + w[0]
            else:
                return None
        result.append(entry)

        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        enlarge_in -= 1

        w = entry
        if enlarge_in == 0:
            enlarge_in = math.pow(2, num_bits)
            num_bits += 1


def decompress_from_encoded_uri(compressed):
    if compressed is None:
        return ""
    if compressed == "":
        return None
    compressed = compressed.replace(" ", "+")
    decompressed = decompress(
        len(compressed),
        32,
        lambda index: get_base_value(KEYSTRURISAFE, compressed[index]),
    )

    return decompressed


def decode_mask(encoded_mask, w, h):

    decomp_string = decompress_from_encoded_uri(encoded_mask)
    decomp_encoded = decomp_string.encode()
    rle_obj = {
        "size": [h, w],
        "counts": decomp_encoded,
    }

    output = mask_utils.decode(rle_obj)
    return output
