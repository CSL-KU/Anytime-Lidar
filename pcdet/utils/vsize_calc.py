import math
#import numba

#@numba.jit(nopython=True)
def calc_area_and_pillar_sz(desired_grid_l : int, area_min_l_cm : int, area_max_l_cm : int):
    if area_min_l_cm % desired_grid_l == 0:
        pillar_size = area_min_l_cm // desired_grid_l
    else:
        pillar_size = area_min_l_cm // desired_grid_l + 1

    area_tmp_l = pillar_size * desired_grid_l

    found = False
    while area_tmp_l >= area_min_l_cm and area_tmp_l <= area_max_l_cm:
        # Before printing, make it meters
        area_l_half = area_tmp_l / 2000.0
        pillar_size_f = pillar_size / 1000.0

        # Make sure the floating point error does not affect grid size
        calc_grid_l = int((area_l_half - (-area_l_half)) / pillar_size_f)

        if calc_grid_l == desired_grid_l:
            found = True
            break

        pillar_size = area_tmp_l // desired_grid_l + 1
        area_tmp_l += desired_grid_l

    if found:
        return -area_l_half, area_l_half, pillar_size_f
    else:
        return 0., 0., 0.

def get_middle_option(max_grid_l, min_grid_l, area_max_l_cm, area_min_l_cm, step):
    options = []
    cur_grid_l = min_grid_l + step
    while cur_grid_l < max_grid_l:
        x = calc_area_and_pillar_sz(cur_grid_l, area_min_l_cm, area_max_l_cm)
        options.append(x)
        cur_grid_l += step
    return options[len(options)//2]


def interpolate_pillar_sizes(max_grid_l, res_divs, pc_range, step=32):
    grid_lens = [int(max_grid_l / rd) for rd in res_divs]
    area_l_cm = int((pc_range[3] - pc_range[0]) * 1000)
    area_min_l_cm = area_l_cm - 1500 #(area_l_cm % 1000)
    area_max_l_cm = area_l_cm + 1500

    all_pc_ranges = []
    all_pillar_sizes = []
    all_grid_lens = []
    resdiv_mask = []
    for i in range(len(grid_lens)-1):
        all_grid_lens.append(grid_lens[i])
        resdiv_mask.append(True)

        all_pc_ranges.append(pc_range)
        psize = (area_l_cm // grid_lens[i]) / 1000
        # The 0.2 can be any number since it is ignored
        psize = [psize, psize, 0.2]
        all_pillar_sizes.append(psize)

        area_left_lim, area_right_lim, new_pillar_size = get_middle_option(
                *grid_lens[i:i+2], area_max_l_cm, area_min_l_cm, step)
        if area_left_lim == 0.:
            print('Couldn\'t find middle pillar size!')
            continue
        all_grid_lens.append(int((area_right_lim-area_left_lim)/new_pillar_size))
        resdiv_mask.append(False)

        new_pc_range = [area_left_lim, area_left_lim, pc_range[2],
                area_right_lim, area_right_lim, pc_range[5]]
        all_pc_ranges.append(new_pc_range)

        psize = [new_pillar_size, new_pillar_size, 0.2]
        all_pillar_sizes.append(psize)
    all_grid_lens.append(grid_lens[-1])
    resdiv_mask.append(True)
    all_pc_ranges.append(pc_range)
    psize = (area_l_cm // grid_lens[-1]) / 1000
    psize = [psize, psize, 0.2]
    all_pillar_sizes.append(psize)

    resdivs = [all_grid_lens[0]/gl for gl in all_grid_lens]

    return all_pc_ranges, all_pillar_sizes, all_grid_lens, resdivs, resdiv_mask

