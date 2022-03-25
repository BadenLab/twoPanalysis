# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:32:56 2021

@author: skrem
"""
import timeit
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(dname)
os.chdir(dname)
import settings_init
from twoP_analysis_functions import *
# import settings_transformations

run_analysis = 0
# run_analysis = 1

def main():
    # Plot_activity()
    if run_analysis == 1:
        RF_list = RF_estimates_list(Model_RF_size, 'lines', settings_init.resolution_folder)
        RF_list_cleaned = Discard_junk_data(RF_list)
        RF_ellipses = RF_ellipses_list(RF_list_cleaned)
        areas = ellipse_param_dfs(RF_ellipses)[3]
        eccs = ellipse_param_dfs(RF_ellipses)[2]
        Ys = ellipse_param_dfs(RF_ellipses)[1]
        Xs = ellipse_param_dfs(RF_ellipses)[0]
        RF_size_model = Model_RF_size()
        plot_amps()
        return RF_list_cleaned, RF_ellipses, areas, eccs, Ys, Xs, RF_size_model
    if run_analysis == 0:
        return None

if __name__ == '__main__':
    start = timeit.default_timer()
    a = main()
    stop = timeit.default_timer()
    print('Time (sec.): ', stop - start)
    print('Time (min.): ', (stop - start)/60)