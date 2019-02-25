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
F_p2 = np.array([[5, 0], [0, 5]])

def main():
    fwc_p1_vm = Local_FWC_P1_Simulator(A_p1, d_p1, C_p1, 832) # max 832, min 89
    fwc_p1_vm.DoE_Run(Z=12, M=10)
    #p1_VMOutput = fwc_p1_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, showType="type-2")  #Default
    p1_VMOutput = fwc_p1_vm.VM_Run(lamda_PLS=15, Z=40, M=10, showType="type-2")  # 변조

    fwc_p2_vm = Local_FWC_P2_Simulator(A_p2, d_p2, C_p2, None, 542) #542, #1432
    fwc_p2_vm.DoE_Run(Z=12, M=10, f=None)
    ez_run1 = fwc_p2_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, f=None, showType="type-2")

    ## 기존과 같이 841로 하면 Process #2는 Bad 현상이 일어난다..
    F_p2 = np.array([[5, 0], [5, 0]])  # 조정하게 한다.
    fwc_p2_vm = Local_FWC_P2_Simulator(A_p2, d_p2, C_p2, F_p2, 542) #10, 200000, 2, 841(real), 1676
    fwc_p2_vm.DoE_Run(Z=12, M=10, f=p1_VMOutput) #반영을 한 모델이어야 하지 않을까?
    F_p2 = np.array([[0, 0], [0, 0]])  # 조정하게 한다.
    ez_run2 = fwc_p2_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10, f=p1_VMOutput, showType="None")

    fwc_p2_vm.plt_show4(41, ez_run1, ez_run2)
    print(fwc_p2_vm.metric)


if __name__ == "__main__":
    main()
