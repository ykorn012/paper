#======================================================================================================
# !/usr/bin/env python
# title          : LocalW2W_FWC_Run.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2018-07-31
# version        : v0.8
# usage          : python LocalW2W_FWC_Run.py
# notes          : Reference Paper "Virtual metrology and feedback control for semiconductor manufacturing"
# python_version : v3.5.3
#======================================================================================================

from simulator.Global_FWC_P3_Simulator import Global_FWC_P3_Simulator
import numpy as np
import os

os.chdir("D:/11. Programming/FactoryWideControl/02. FactoryWideControl/")

Tgt_p1 = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

Tgt_p2 = np.array([0, 0])
A_p2 = np.array([[1, 0.1], [-0.5, 0.2]])
d_p2 = np.array([[0, 0.05], [0, 0.05]])
C_p2 = np.transpose(np.array([[0.1, 0, 0, -0.2, 0.1], [0, -0.2, 0, 0.3, 0]]))
F_p2 = np.array([[0.5, 0], [0, 0.5]])


# test = np.array([[0.3, 0], [0, 0.5], [0.4, 0.7]])
# np.max(test[:,1], axis=0)

def main():


    # p1_r2r_VMOutput = np.loadtxt('output/vm_output2.csv', delimiter=',')
    # temp = np.loadtxt('output/vm_output1.csv', delimiter=',')

    # fwc_p2_pre_r2r = Global_FWC_P2_Simulator(Tgt_p2, A_p2, d_p2, C_p2, F_p2, 1000000000)
    # fwc_p2_pre_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.55, dEWMA_Wgt2=0.75, Z=20, M=10, f=p1_r2r_VMOutput, isR2R=True)
    # VM_Output = fwc_p2_pre_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.55, dEWMA_Wgt2=0.75, Z=20, M=10, f=p1_r2r_VMOutput, isR2R=True)

    max_i = 0
    max_value = 10000000
    min_value = 0

    term = 1
    for i in range(term, term + 100):
        print("index : ", i)
        fwc_p2_pre_r2r = Global_FWC_P3_Simulator(Tgt_p2, A_p2, d_p2, C_p2, F_p2, i)  # 10, 200000, 2
        fwc_p2_pre_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.55, dEWMA_Wgt2=0.75, Z=10, M=10, f=None, isR2R=True)
        VM_Output = fwc_p2_pre_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.55, dEWMA_Wgt2=0.75, Z=20, M=10, f=None, isR2R=True)
        cri = np.max(np.absolute(VM_Output[:,1]), axis=0)
        print("cri : ", cri)

        if cri > min_value:
            max_i = i
            min_value = cri

    print("max_i = ", max_i)
    print("max_value = ", min_value)

if __name__ == "__main__":
    main()
