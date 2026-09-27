#!/usr/bin/env python3
"""Extract the MTP head of a qwen35moe GGUF into a standalone draft model.

usage: extract_mtp_head.py SOURCE.gguf OUTPUT.gguf

The draft keeps the source metadata and copies the nextn layer's tensors plus the tensors the loader
requires unconditionally, byte for byte, so the quantized types are preserved.
"""
import sys

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGUFValueType

# the nextn layer of the models this fork supports
MTP_LAYER = 40

# required regardless of the mtp-only path, see load_arch_tensors
REQUIRED_GLOBALS = ("token_embd.weight", "output_norm.weight")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    src_path, dst_path = sys.argv[1], sys.argv[2]

    reader = GGUFReader(src_path)
    arch = str(reader.fields["general.architecture"].contents())

    writer = GGUFWriter(dst_path, arch, use_temp_file=False)

    # the reader exposes the header as GGUF.* fields and the writer emits its own, so they are not copied
    skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}

    for key, field in reader.fields.items():
        if key in skip:
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)

    prefix = f"blk.{MTP_LAYER}."
    n_copied = 0
    for tensor in reader.tensors:
        if not (tensor.name.startswith(prefix) or tensor.name in REQUIRED_GLOBALS):
            continue
        # tensor.shape is the file dims, tensor.data.shape is the byte shape the writer expects
        writer.add_tensor(tensor.name, np.asarray(tensor.data), raw_shape=list(tensor.data.shape),
                          raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
        n_copied += 1

    print(f"copied {n_copied} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
