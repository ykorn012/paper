#======================================================================================================
# !/usr/bin/env python
# title          : Global_FWC_P1_Simulator.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2018-07-31
# version        : v0.8
# usage          : python GlobalW2W_FWC_Run.py
# notes          : Reference Paper "An Approach for Factory-Wide Control Utilizing Virtual Metrology"
# python_version : v3.5.3
#======================================================================================================
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics

class Global_FWC_P3_Simulator:

    def __init__(self, Tgt, A, d, C, F, seed):
        self.pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)
        np.random.seed(seed)
        self.Tgt = Tgt
        self.A = A
        self.d = d
        self.C = C
        self.F = F

    def sampling_vp(self):
        v1 = np.random.normal(-0.4, np.sqrt(0.2))
        v2 = 2 * v1
        v3 = np.random.uniform(0.2, 0.6)
        v4 = 3 * v3
        v5 = np.random.uniform(0, 0.4)

        v = np.array([v1, v2, v3, v4, v5])
        return v

    def sampling_ep(self):
        e1 = np.random.normal(0, np.sqrt(0.05))
        e2 = np.random.normal(0, np.sqrt(0.1))
        e = np.array([e1, e2])
        return e

    def sampling(self, k, uk=np.array([0, 0]), vp=np.array([0, 0, 0, 0, 0]), ep=np.array([0, 0]), fp=np.array([0, 0]), isInit=True):
        u1 = uk[0]
        u2 = uk[1]
        u = uk

        v1 = vp[0]
        v2 = vp[1]
        v3 = vp[2]
        v4 = vp[3]
        v5 = vp[4]

        v = vp
        e = ep

        k1 = k
        k2 = k
        eta_k = np.array([[k1], [k2]])

        psi = np.array([u1, u2, v1, v2, v3, v4, v5, k1, k2])

        if fp is not None:
            psi = np.r_[psi, fp]
            f = fp
            y = u.dot(self.A) + v.dot(self.C) + np.sum(eta_k * self.d, axis=0) + f.dot(self.F) + e
        else:
            y = u.dot(self.A) + v.dot(self.C) + np.sum(eta_k * self.d, axis=0) + e

        rows = np.r_[psi, y]

        idx_end = len(rows)
        idx_start = idx_end - 2
        return idx_start, idx_end, rows  #y값의 시작과 끝 정보, 전체 값 정보

    def pls_update(self, V, Y):
        self.pls.fit(V, Y)
        return self.pls

    def setDoE_Mean(self, DoE_Mean):
        self.DoE_Mean = DoE_Mean

    def getDoE_Mean(self):
        return self.DoE_Mean

    def setPlsWindow(self, PlsWindow):
        self.PlsWindow = PlsWindow

    def getPlsWindow(self):
        return self.PlsWindow

    def plt_show1(self, n, y_act):
        plt.figure()
        plt.plot(np.arange(1, n + 1), y_act, 'bx--', lw=2, ms=10, mew=2)
        plt.xticks(np.arange(0, n + 1, 20))
        plt.xlabel('Run No.')
        plt.ylabel('Actual Response (y2)')

    def plt_show2(self, n, y_act):
        plt.plot(np.arange(1, n + 1), y_act, 'ro-', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 20))
        plt.xlabel('Run No.')
        plt.ylabel('Actual Response (y2)')

    def DoE_Run(self, lamda_PLS, dEWMA_Wgt1, dEWMA_Wgt2, Z, M, f, isR2R):
        N = Z * M
        I = np.identity(2)
        dEWMA_Wgt1 = dEWMA_Wgt1 * I
        dEWMA_Wgt2 = dEWMA_Wgt2 * I
        DoE_Queue = []

        sample_init_VP = []
        sample_init_EP = []

        for k in range(0, N + 1):
            sample_init_VP.append(self.sampling_vp())
            sample_init_EP.append(self.sampling_ep())
        vp_next = sample_init_VP[0]
        ep_next = sample_init_EP[0]

        for k in range(1, N + 1):  # range(101) = [0, 1, 2, ..., 100])
            if f is not None:
                fp = f[k - 1,0:2]
                p1_lamda_PLS = f[k - 1, 2:3]
                fp = p1_lamda_PLS * fp
                if k == 1:
                    uk_next = np.array([-54.48, -108.8])  # 계산 공식에 의해
                    Dk_prev = np.array([-0.067, 26.5])
                    Kd_prev = np.array([0.16, 0.62])
            else:
                fp = None
                if k == 1:
                    uk_next = np.array([0, 0])  # 계산 공식에 의해
                    Dk_prev = np.array([0, 0])
                    Kd_prev = np.array([0, 0])

            idx_start, idx_end, result = self.sampling(k, uk_next, vp_next, ep_next, fp, True)
            npResult = np.array(result)

            #================================== initVM-R2R Control =====================================
            uk = npResult[0:2]
            yk = npResult[idx_start:idx_end]

            Dk = (yk - uk.dot(self.A)).dot(dEWMA_Wgt1) + Dk_prev.dot(I - dEWMA_Wgt1)
            Kd = (yk - uk.dot(self.A) - Dk_prev).dot(dEWMA_Wgt2) + Kd_prev.dot(I - dEWMA_Wgt2)

            Kd_prev = Kd
            Dk_prev = Dk

            if isR2R == True:
                uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                vp_next = sample_init_VP[k]

            else:
                if k % M == 0:
                    uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                    vp_next = sample_init_VP[k]
            ep_next = sample_init_EP[k]
            DoE_Queue.append(result)

        initplsWindow = DoE_Queue.copy()
        npPlsWindow = np.array(initplsWindow)

        plsWindow = []

        #np.savetxt("output/npPlsWindow1.csv", npPlsWindow, delimiter=",", fmt="%s")

        if f is not None:
            for k in range(0, N):  # range(101) = [0, 1, 2, ..., 100])
                p1_lamda_PLS = f[k,2:3]
                if (k + 1) % M != 0:
                    npPlsWindow[k, idx_start - 2:idx_start] = p1_lamda_PLS * npPlsWindow[k, idx_start - 2:idx_start]

        for z in np.arange(0, Z):
            npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start] = lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start]
            npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end] = lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end])

        for i in range(len(npPlsWindow)):
            plsWindow.append(npPlsWindow[i])

        #np.savetxt("output/npPlsWindow2.csv", npPlsWindow, delimiter=",", fmt="%s")

        npDoE_Queue = np.array(plsWindow)
        DoE_Mean = np.mean(npDoE_Queue, axis=0)

        plsModelData = npDoE_Queue - DoE_Mean
        V0 = plsModelData[:, 0:idx_start]
        Y0 = plsModelData[:, idx_start:idx_end]

        pls = self.pls_update(V0, Y0)

        # print('Init VM Coefficients: \n', pls.coef_)
        y_pred = pls.predict(V0) + DoE_Mean[idx_start:idx_end]
        y_act = npDoE_Queue[:, idx_start:idx_end]

        # print("Init VM Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
        # print("Init VM r2 score: %.3f" % metrics.r2_score(y_act, y_pred))


        self.setDoE_Mean(DoE_Mean)
        self.setPlsWindow(plsWindow)
        # self.plt_show2(N, y_act[:, 1:2])
        # self.plt_show1(N, y_pred[:, 1:2])


    def VM_Run(self, lamda_PLS, dEWMA_Wgt1, dEWMA_Wgt2, Z, M, f, isR2R):
        N = Z * M
        I = np.identity(2)
        dEWMA_Wgt1 = dEWMA_Wgt1 * I
        dEWMA_Wgt2 = dEWMA_Wgt2 * I

        ## V0, Y0 Mean Center
        DoE_Mean = self.getDoE_Mean()
        idx_end = len(DoE_Mean)
        idx_start = idx_end - 2
        meanVz = DoE_Mean[0:idx_start]
        meanYz = DoE_Mean[idx_start:idx_end]
        yk = np.array([0, 0])

        Dk_prev = np.array([-0.067, 26.5])  # 10번째 run시 값
        Kd_prev = np.array([0.16, 0.62])  # 10번째 run시 값

        # Dk = np.array([0, 0])
        # Kd = np.array([0, 0])

        uk_next = np.array([-54.48, -108.8])  # 계산 공식에 의해

        M_Queue = []
        ez_Queue = []
        ez_Queue.append([0, 0])
        y_act = []
        y_pred = []
        VM_Output = []

        plsWindow = self.getPlsWindow()

        sample_vm_VP = []
        sample_vm_EP = []

        for k in range(0, N + 1):
            sample_vm_VP.append(self.sampling_vp())
            sample_vm_EP.append(self.sampling_ep())
        vp_next = sample_vm_VP[0]
        ep_next = sample_vm_EP[0]

        for z in np.arange(0, Z):
            for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
                if f is not None:
                    fp = f[k - 1, 0:2]
                else:
                    fp = None
                    if k == 1:
                        uk_next = np.array([0, 0])  # 계산 공식에 의해
                        Dk_prev = np.array([0, 0])
                        Kd_prev = np.array([0, 0])

                # y값의 시작과 끝 정보, 전체 값 정보
                idx_start, idx_end, result = self.sampling(k, uk_next, vp_next, ep_next, fp, False)
                psiK = result[0:idx_start]  # 파라미터 값들
                psiKStar = psiK - meanVz  # 파라미터 값들 평균 마이너스
                y_predK = self.pls.predict(psiKStar.reshape(1, idx_start)) + meanYz   # 예측값 + 평균
                rows = np.r_[result, y_predK.reshape(2, )]   #실제값 + 2개 예측값을 rows로, run 10일때가 actual, vm 차이 비교

                y_pred.append(rows[idx_end:idx_end + 2])  #예측 값  ==> 10개의 VM 값인데..
                y_act.append(rows[idx_start:idx_end])     #실제 값   ==> 시뮬레이션의 실제 값 인데..

                # ================================== VM + R2R Control =====================================
                if k % M != 0:   #예측 값
                    yk = rows[idx_end:idx_end + 2]
                else:
                    yk = rows[idx_start:idx_end]    #실제 값
                    e1 = np.absolute(rows[idx_start + 1:idx_end] - rows[idx_end + 1:idx_end + 2])
                uk = psiK[0:2]

                Dk = (yk - uk.dot(self.A)).dot(dEWMA_Wgt1) + Dk_prev.dot(I - dEWMA_Wgt1)
                Kd = (yk - uk.dot(self.A) - Dk_prev).dot(dEWMA_Wgt2) + Kd_prev.dot(I - dEWMA_Wgt2)

                Kd_prev = Kd
                Dk_prev = Dk

                if isR2R == True:
                    uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                    vp_next = sample_vm_VP[k]

                uk_next = uk_next.reshape(2, )
                ep_next = sample_vm_EP[k]

                M_Queue.append(rows)  # M_Queue에 rows의 정보

            del plsWindow[0:M]   #Queue의 가장 처음 Run 10이 없어진다.

            if isR2R == False:
                uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                vp_next = sample_vm_VP[k]

            # 여기서 부터는 모델 업데이트를 위한 과정이다. 이미 VM은 rows 정보에 있지만, 가중치를 반영해 준다.

            if z == 0:
                ez = 0
            npM_Queue = np.array(M_Queue)  #parameter + 실제값 + 2개 예측값을 rows로, run 10일 때가 actual, vm 차이 비교
            # M은 Run 주기이며, 10, M-1은 run = 10을 제외한 VM들이겠지
            # idx_start 까지는 파라미터 값들로 lamda_PLS 0.1을 반영하겠다는 의미지..

            for i in range(M):  #VM_Output 구한다. lamda_pls 가중치를 반영하지 않는다.
                if i == M - 1:
                    temp = npM_Queue[i:i + 1, idx_start:idx_end]
                else:
                    temp = npM_Queue[i:i + 1, idx_end:idx_end + 2]
                VM_Output.append(np.array([temp[0, 0], temp[0, 1]]))

            # emax = 5
            # lamda_PLS = 1 - e1/emax
            # if lamda_PLS <= 0:
            #     lamda_PLS = 0.1
            #
            # print("e1 : ", e1, "P2 lamda_PLS : ", lamda_PLS)

            if f is not None:
                p1_lamda_PLS = f[k - 1, 2:3]
                npM_Queue[0:M - 1, idx_start - 2:idx_start] = p1_lamda_PLS * npM_Queue[0:M - 1, idx_start - 2:idx_start]

            #np.savetxt("output/npM_Queue2.csv", npM_Queue, delimiter=",", fmt="%s")

            npM_Queue[0:M - 1, 0:idx_start] = lamda_PLS * npM_Queue[0:M - 1, 0:idx_start]
            # idx_start:idx_end는 실제 값에 VM 값들의 조정을 통해 모델을 위해 VM의 정보를 업데이트 한다.
            npM_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * (npM_Queue[0:M - 1, idx_end:idx_end + 2] + 0.5 * ez)
            #npM_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * (npM_Queue[0:M - 1, idx_end:idx_end + 2])  # 0.5 * ez 반영안할시
            npM_Queue = npM_Queue[:, 0:idx_end] #여기에는 VM + Actual 실제값들이 저장되어 있다.

            # for i in range(M):  #VM_Output 구한다. lamda_pls 가중치를 반영하지 않는다.
            #     temp = npM_Queue[i:i + 1, idx_start:idx_end]
            #     VM_Output.append(np.array([temp[0, 0], temp[0, 1]]))

            for i in range(M):
                plsWindow.append(npM_Queue[i])  #전체 Queue에 넣는다.

            M_Mean = np.mean(plsWindow, axis=0)  #Queue의 평균을 구한다.
            meanVz = M_Mean[0:idx_start]   #파라미터 평균
            meanYz = M_Mean[idx_start:idx_end]  #y값 run시마다 vm 9개(lamda_pla 0.1) 실제 1개(lamda_pls 1) 평균

            plsModelData = plsWindow - M_Mean   #Queue의 평균 제외
            V = plsModelData[:, 0:idx_start]    #모델을 위한 파라미터
            Y = plsModelData[:, idx_start:idx_end]   #모델을 위한 y값

            self.pls_update(V, Y)
            ez = M_Queue[M - 1][idx_start:idx_end] - M_Queue[M - 1][idx_end:idx_end + 2]
            ez_Queue.append(ez)
            # print("ez : ", ez)

            del M_Queue[0:M]

        y_act = np.array(y_act)
        y_pred = np.array(y_pred)

        # print("VM Mean squared error: %.3f" % metrics.mean_squared_error(y_act[:,1:2], y_pred[:,1:2]))
        # print("VM r2 score: %.3f" % metrics.r2_score(y_act[:,1:2], y_pred[:,1:2]))
        return y_act
