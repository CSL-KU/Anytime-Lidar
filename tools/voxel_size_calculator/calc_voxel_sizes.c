#include <stdio.h>
#include <math.h>

// Everything in centimeters!

int main(int argc, char* argv[])
{
	int voxel_size, area_tmp_l;
	int area_min_l = 102000, area_max_l = 103000;
	int min_grid_l = 384, max_grid_l = 1408;
	int cur_grid_l = min_grid_l;
	printf("Area_length,voxel_size\n");
	while(cur_grid_l <= max_grid_l){
		// Find all area length multiple of grid length
		printf("For grid size %d x %d:\n", cur_grid_l, cur_grid_l);
		if(area_min_l % cur_grid_l == 0)
			voxel_size = area_min_l / cur_grid_l;
		else
			voxel_size = area_min_l / cur_grid_l + 1;
		area_tmp_l = voxel_size * cur_grid_l;
		while(area_tmp_l >= area_min_l && area_tmp_l <= area_max_l){
			// Before printing, make it meters
			double area_l_half = area_tmp_l / 2000.0;
			double voxel_size_f = voxel_size / 1000.0;
			// Make sure the floating point error does not affect grid size
			int calc_grid_l = (area_l_half - (-area_l_half)) / voxel_size_f;
			if (calc_grid_l == cur_grid_l)
			{
				printf("\tArea lims (m): -%f %f Voxel length (m): %f\n",
						area_l_half, area_l_half, voxel_size_f);
			}
			voxel_size = area_tmp_l / cur_grid_l + 1;
			area_tmp_l += cur_grid_l;
		}

		cur_grid_l += 64;
	}

	return 0;
}
