import numba
import numpy as np

class SchedAlgo:
    ProjectionOnly = 0
    RoundRobin = 2
    AdaptiveRR = 3
    RoundRobin_NoProj = 4
    # 5 and 6 are other baselines 01 and pillar
    RoundRobin_NoSchedNoProj = 7
    # 8 is VoxelNeXt
    RoundRobin_VN = 9  # VoxelNeXt
    RoundRobin_16 = 10  # Same as RoundRobin, 16 tiles instead of 18
    # Keep the next ones only for code compability
    MirrorRR = 98

@numba.jit(nopython=True)
def get_num_tiles(ctc): # chosen tile coords
    # it could be contiguous or not, detect that
    ctc_s, ctc_e = ctc[0], ctc[-1]
    if ctc_s <= ctc_e:
        num_tiles = ctc_e - ctc_s + 1
    else:
        j = 0
        while ctc[j] < ctc[j+1]:
            j += 1
        num_tiles = ctc[j] - ctc_s + 1 + ctc_e - ctc[j+1] + 1

    return num_tiles

@numba.jit(nopython=True)
def fill_tile_gaps(netc, vcounts):
    num_tiles = netc[-1] - netc[0] + 1
    if num_tiles == netc.shape[0]:
        return netc, vcounts  # no need
    else:
        new_netc = np.arange(netc[0], netc[-1] + 1, dtype=netc.dtype)
        new_vcounts= np.zeros(num_tiles, dtype=vcounts.dtype)
        i, j = 0, 0
        while i < netc.shape[0]:
            if netc[i] == new_netc[j]:
                new_vcounts[j] = vcounts[i]
                i+=1
            j+=1
    return new_netc, new_vcounts

@numba.jit(nopython=True)
def round_robin_sched_helper(netc, last_tile_coord, tcount):
    num_nonempty_tiles = netc.shape[0]
    tile_begin_idx=0
    for i in range(num_nonempty_tiles):
        if netc[i] > last_tile_coord:
            tile_begin_idx = i
            break

    netc_flip = np.concatenate((netc[tile_begin_idx:], netc[:tile_begin_idx]))

    num_tiles = np.empty((num_nonempty_tiles,), dtype=np.int32)
    for i in range(netc_flip.shape[0]):
        ctc = netc_flip[:i+1]
        num_tiles[i] = get_num_tiles(ctc)

    return num_tiles, netc_flip

@numba.jit(nopython=True)
def mirror_sched_helper(netc, netc_vcounts, last_tile_coord, tcount):
    m2 = tcount//2
    m1 = m2 - 1
    mtiles = np.array([m1, m2], dtype=netc.dtype)
    rtiles = np.arange(m2+1, netc[-1]+1)
    ltiles = np.arange(m1-1, netc[0]-1, -1)
    rltiles = np.concatenate((rtiles, ltiles, rtiles, ltiles))

    reference_tiles = np.concatenate((np.arange(m2+1, tcount), np.arange(m1-1, -1, -1)))
    reference_tiles = np.concatenate((reference_tiles, reference_tiles))

    idx = 0
    while reference_tiles[idx] != last_tile_coord:
        idx += 1
    idx += 1
    start_tile = reference_tiles[idx]
    while (start_tile not in rtiles) and (start_tile not in ltiles):
        idx += 1
        start_tile = reference_tiles[idx]

    idx, = np.where(rltiles == start_tile)
    rltiles = rltiles[idx[0]:idx[0]+netc.shape[0]-2]

    nv = np.zeros((tcount,), dtype=netc_vcounts.dtype)
    nv[netc] = netc_vcounts

    vcounts_all = np.zeros((netc.shape[0]-mtiles.shape[0]+1, tcount), dtype=np.float32)
    num_tiles = np.empty((vcounts_all.shape[0],), dtype=np.int32)

    #vcounts[0] represents running mandatory only
    for i in range(vcounts_all.shape[0]):
        ctc = np.concatenate((mtiles, rltiles[:i]))
        num_tiles[i] = ctc.shape[0]
        for j in range(num_tiles[i]):
            vcounts_all[i, ctc[j]] = nv[ctc[j]]

    return num_tiles, vcounts_all, rltiles

