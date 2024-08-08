def timeloop_arch_cb(spec, einsum, PE_dim, multiplier, E):
    min_tiles, l3_sz = get_l3_sz(PE_dim, multiplier, E)
    if einsum in {"QK", "LM", "SLN", "SLD", "SLNV"}:
        spec["architecture"]["nodes"].find("PE_col")["spatial"]["meshX"] = PE_dim
        spec["architecture"]["nodes"].find("PE")["spatial"]["meshY"] = PE_dim

        spec["architecture"]["nodes"].find("reg_file_1d")["attributes"]["depth"] = PE_dim * 8 * min_tiles
        spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = 4 + min_tiles
    else:
        spec["architecture"]["nodes"].find("PE")["spatial"]["meshX"] = PE_dim
        spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = 8 * min_tiles

    # Width = PE_dim values = PE_dim * 16 bits
    spec["architecture"]["nodes"].find("L3")["attributes"]["width"] = PE_dim * 16
    spec["architecture"]["nodes"].find("L3")["attributes"]["depth"] = l3_sz // (PE_dim * 2)

def accelergy_arch_cb(spec, array, PE_dim, multiplier, E):
    min_tiles, l3_sz = get_l3_sz(PE_dim, multiplier, E)
    if array == "2d":
        # Width = 256 values = 512 bits
        spec["architecture"]["nodes"].find("global_buffer")["attributes"]["width"] = PE_dim * 16
        spec["architecture"]["nodes"].find("global_buffer")["attributes"]["depth"] = l3_sz // (PE_dim * 2)

        spec["architecture"]["nodes"].find("PE_col")["spatial"]["meshX"] = PE_dim
        spec["architecture"]["nodes"].find("PE")["spatial"]["meshY"] = PE_dim

        spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = 2 * min_tiles

    elif array == "1d":
        spec["architecture"]["nodes"].find("PE")["spatial"]["meshX"] = PE_dim
        spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = 8 * min_tiles
    else:
        raise NotImplementedError

def get_l3_sz(PE_dim, multiplier, E):
    min_tiles = max(1, PE_dim // E) * 2
    num_tiles = 256 * 2**10 // PE_dim

    min_P1 = max(1, min_tiles // num_tiles)

    # PE_dim * min_P1 * E/F * num_fibers * 2B
    min_l3 = PE_dim * min_P1 * E * 8 * 2

    return min_tiles, min_l3 * multiplier

