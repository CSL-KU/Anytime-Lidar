import json
import sys
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ALv2_res_path = './streaming_eval_ALv2'
BL_res_path = './streaming_eval_baseline'

def get_data_dict(results_path, budgets_str, e2e_deadlines_str):
    data_dict = {e2e_dl:{b:list() for b in budgets_str} for e2e_dl in e2e_deadlines_str}
    paths = glob.glob(results_path + '/*')
    for path in paths:
        with open(path, 'r') as handle:
            eval_dict= json.load(handle)
        e2e_dl= str(round(float(path.split('/')[-1].split('_')[0][1:]),3))
        if e2e_dl in data_dict:
            budget = str(round(eval_dict['deadline_sec'],3))
            if budget in data_dict[e2e_dl]:
                mAP = eval_dict['eval_results_dict']['mAP']
                NDS = eval_dict['eval_results_dict']['NDS']
                e2e_times_ms = eval_dict['exec_times']['End-to-end'] 
                total_e2e_time_ms = sum(e2e_times_ms)
                #utilization = total_e2e_time_ms / (1961 * 50.)
                utilization = total_e2e_time_ms / (2000 * 50.)

                data_dict[e2e_dl][budget].extend([mAP, NDS, utilization])
    return data_dict

max_mAP=0.65
budgets = np.arange(0.180, 0.241, 0.020)
e2e_deadlines = np.arange(0.200, 0.501, 0.100)
budgets_str = [str(round(b,3)) for b in budgets]
e2e_deadlines_str = [str(round(e2e_dl,3)) for e2e_dl in e2e_deadlines]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
for ax, idx, ylabel, div, ylim in zip (axes, (0, 2), ('Normalized accuracy(%)', 'Utilization(%)'), \
        (max_mAP*0.01, 0.01), ((0., 50.),(0.,100.))):
    data_dict = get_data_dict(ALv2_res_path, budgets_str, e2e_deadlines_str)
    for budget in budgets_str:
        mask = (e2e_deadlines >= float(budget)-0.001)
        mAP_l = [data_dict[str(round(e2e_dl,3))][budget][idx]/div for e2e_dl in e2e_deadlines[mask]]
        xvals = (e2e_deadlines[mask]*1000).astype(int)
        l2d = ax.plot(xvals, mAP_l)
        b = int(float(budget)*1000)
        ax.scatter(xvals, mAP_l,
                label=f'ALv2 {b} ms budget', color=l2d[0].get_c())
    bl_budgets_str = ['10.0']
    data_dict = get_data_dict(BL_res_path, bl_budgets_str, e2e_deadlines_str)
    for budget in bl_budgets_str:
        #mask = (e2e_deadlines > float(budget)-0.001)
        mAP_l = [data_dict[str(round(e2e_dl,3))][budget][idx]/div for e2e_dl in e2e_deadlines]
        xvals = (e2e_deadlines*1000).astype(int)
        l2d = ax.plot(xvals, mAP_l)
        ax.scatter(xvals, mAP_l, label=f'CenterPoint', color=l2d[0].get_c())

    ax.invert_xaxis()
    ax.set_ylabel(ylabel, fontsize='x-large')
    ax.set_xlabel('Period (sec)', fontsize='x-large')
    ax.legend(ncol=2)
    ax.set_ylim(ylim)
plt.savefig("/root/shared_data/stream_eval.png")
