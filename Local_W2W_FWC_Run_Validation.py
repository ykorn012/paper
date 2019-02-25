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
from simulator.Local_FWC_P1_Simulator import Local_FWC_P1_Simulator
from simulator.Local_FWC_P2_Simulator import Local_FWC_P2_Simulator
import numpy as np

A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

A_p2 = np.array([[1, 0.1], [-0.5, 0.2]])
d_p2 = np.array([[0, 0.05], [0, 0.05]])
C_p2 = np.transpose(np.array([[0.1, 0, 0, -0.2, 0.1], [0, -0.2, 0, 0.3, 0]]))
F_p2 = np.array([[0.5, 0], [0, 0.5]])

def main():
    # fwc_p1_vm = Local_FWC_P1_Simulator(A_p1, d_p1, C_p1, 832)
    # fwc_p1_vm.DoE_Run(Z=12, M=10)
    # p1_VMOutput = fwc_p1_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, showType="None")

    # fwc_p2_vm = Local_FWC_P2_Simulator(A_p2, d_p2, C_p2, None, 542)
    # fwc_p2_vm.DoE_Run(Z=12, M=10, f=None)
    # ez_run1 = fwc_p2_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, f=None, showType="None")

    # fwc_p2_vm = Local_FWC_P2_Simulator(A_p2, d_p2, C_p2, F_p2, 2) #10, 200000, 2, 841
    # fwc_p2_vm.DoE_Run(Z=12, M=10, f=p1_VMOutput)
    # ez_run2 = fwc_p2_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, f=p1_VMOutput, showType="None")
    #
    # fwc_p2_vm.plt_show2(41, ez_run1, ez_run2)
    # print(fwc_p2_vm.metric)
    #



    max_value = 0
    min_value = 10000000
    for i in range(1000):
        print("index : ", i)
        fwc_p1_vm = Local_FWC_P1_Simulator(A_p1, d_p1, C_p1, i)
        fwc_p1_vm.DoE_Run(Z=12, M=10)
        p1_VMOutput = fwc_p1_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, showType="None")
        if fwc_p1_vm.metric > max_value:
            max_i = i
            max_value = fwc_p1_vm.metric
        if fwc_p1_vm.metric < min_value:
            min_i = i
            min_value = fwc_p1_vm.metric

    # print("max_i = ", max_i)
    # print("max_value = ", max_value)
    # print("min_i = ", min_i)
    # print("min_value = ", min_value)

    max_value = 0
    min_value = 10000000
    for i in range(0, 100):
        print("index : ", i)
        fwc_p2_vm = Local_FWC_P2_Simulator(A_p2, d_p2, C_p2, F_p2, i)  # 10, 200000, 2
        fwc_p2_vm.DoE_Run(Z=12, M=10, f=p1_VMOutput)
        ez_run2 = fwc_p2_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, f=p1_VMOutput, showType="None")
        if fwc_p2_vm.metric > max_value:
            max_i = i
            max_value = fwc_p2_vm.metric
        if fwc_p2_vm.metric < min_value:
            min_i = i
            min_value = fwc_p2_vm.metric

    print("max_i = ", max_i)
    print("max_value = ", max_value)
    print("min_i = ", min_i)
    print("min_value = ", min_value)
if __name__ == "__main__":
    main()
