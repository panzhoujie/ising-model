# This is a sample Python script.
import random

import numpy
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import math
from random import *
import matplotlib.pyplot as plt
import cv2
import copy
from pbit import func
  # Press Ctrl+F8 to toggle the breakpoint.


p_reco = list(numpy.loadtxt('p_num3.txt'))
step_reco = list(numpy.loadtxt('v_num3.txt'))

p_reco.append(1)
step_reco.append(4.3)

p_reco.append(0)
step_reco.append(3.2)

p_reco.append(1)
step_reco.append(4.4)

p_reco.append(0)
step_reco.append(3.1)

p_reco.append(0)
step_reco.append(3.0)

p_reco.append(1)
step_reco.append(4.5)
class max_cut():
    def __init__(self):
        self.X = np.array([[1, 0, 0, 0, 1],[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],[0,1,0,1,0],[1,0,0,0,1]])
        self.Y = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        self.B = 4
        self.diy = None

    def neighbour(self, x, y):
        lx = [-1,-1,1,1]
        ly = [-1,1,-1,1]
        rx =[]
        ry =[]
        if x==0 and y==2:
            rx.append(0)
            ry.append(0)
            rx.append(0)
            ry.append(4)
        elif x==2 and y==0:
            rx.append(0)
            ry.append(0)
            rx.append(4)
            ry.append(0)
        elif x==4 and y==2:
            rx.append(4)
            ry.append(0)
            rx.append(4)
            ry.append(4)
        elif x==2 and y==4:
            rx.append(0)
            ry.append(4)
            rx.append(4)
            ry.append(4)
        else:
            for i in range(0, 4):
                newx = x + lx[i]
                newy = y + ly[i]
                if newx>4:
                    newx = 0
                elif newx<0:
                    newx = 4
                if newy>4:
                    newy = 0
                elif newy<0:
                    newy = 4
                rx.append(newx)
                ry.append(newy)
        return rx, ry

    def edge_query(self, x1, y1, x2, y2, mode):
        if mode == 'x':
            judge = (self.X[x1,y1]+1)%2
            if judge == self.X[x2,y2]:
                return 1
            else:
                return -1
        elif mode =='y':
            judge = (self.Y[x1, y1] + 1) % 2
            if judge == self.Y[x2, y2]:
                return 1
            else:
                return -1
        else:
            judge = (self.diy[x1, y1] + 1) % 2
            if judge == self.diy[x2, y2]:
                return 1
            else:
                return -1

    def spinlize(self, value):
        v = 1
        if value == 0:
            v = -1
        return v

    def single_step(self, cm, x, y, mode):
        rx, ry = self.neighbour(x, y)
        lenx = len(rx)
        deltaE = 0
        for i in range(0, lenx):
            deltaE = deltaE + (self.edge_query(x, y, rx[i], ry[i], mode) * self.spinlize(cm[rx[i],ry[i]])
                      * (self.spinlize(cm[x,y])-self.spinlize((cm[x,y]+1)%2)))
        p = random()
        if p < (math.exp(self.B * deltaE)-0.5):
            cm[x, y] = (cm[x, y]+1) % 2 # flip

    def diy_set(self, DIY):
        self.diy = DIY

    def neighbourdiy(self, x, y):
        sx = (self.diy).shape[0]
        sy = (self.diy).shape[1]
        lx = [-1,-1,-1,0,0,0,1,1,1]
        ly = [-1,0,1,-1,0,1,-1,0,1]
        rx = []
        ry = []
        for i in range(0, 9):
            newx = x + lx[i]
            newy = y + ly[i]
            if newx >= sx:
                newx = newx - sx
            elif newx < 0:
                newx = sx + newx
            if newy >= sy:
                newy = newy - sy
            elif newy < 0:
                newy = sy + newy
            rx.append(newx)
            ry.append(newy)
        return rx, ry

    def fullE(self, cm, mode):
        deltaE = 0
        if mode =='x' or mode == 'y':
            for x in range(0, 5):
                for y in range(0, 5):
                    rx, ry = self.neighbour(x, y)
                    lenx = len(rx)
                    for i in range(0, lenx):
                        deltaE = deltaE + self.edge_query(x, y, rx[i], ry[i], mode) * self.spinlize(cm[rx[i], ry[i]]) * self.spinlize(cm[x, y])
            deltaE = deltaE/2
        else:
            sx = (self.diy).shape[0]
            sy = (self.diy).shape[1]
            for x in range(0, sx):
                for y in range(0, sy):
                    rx, ry = self.neighbourdiy(x, y)
                    lenx = len(rx)
                    for i in range(0, lenx):
                        deltaE = deltaE + self.edge_query(x, y, rx[i], ry[i], mode) * self.spinlize(cm[rx[i], ry[i]]) * self.spinlize(cm[x, y])
            deltaE = deltaE/2
        return deltaE

    def simulate(self, mode, max_step=1000):
        cm = np.random.randint(0, 2, size=(5, 5))
        plt.ion()
        E = self.fullE(cm, mode)
        E_reco = [E]
        plt.imshow(cm)
        plt.pause(1)
        plt.clf()
        step_reco = [0]
        for count in range(0, max_step):
            randx = randint(0, 4)
            randy = randint(0, 4)
            self.single_step(cm, randx, randy, mode)
            if count % 10 == 0:
                E_reco.append(self.fullE(cm, mode))
                step_reco.append(count)
                plt.imshow(cm)
                plt.pause(1)
                plt.clf()
        plt.ioff()
        fig = plt.figure()
        plt.plot(step_reco, E_reco)
        plt.show()

class mem_maxcut_sim(max_cut):
    def __init__(self):
        super(mem_maxcut_sim, self).__init__()
        self.ntemp = None
        self.temp = None
        self.nmem = None
        self.mem = None # 0:HRS, 1:LRS
        self.onoffratio = 20 # LRS=1, HRS/LRS
        self.Rf = 0.1   # 跨阻放大器 反馈电阻/LRS
        self.Vread = 0.1

    def defineproblem(self, mode):
        if mode=='x':
            s = (self.X).shape[0]
            self.mem = np.zeros((s, s))
            self.nmem = np.ones((s, s))  # 定义(spin)与(nspin)
        elif mode=='y':
            s = (self.Y).shape[0]
            self.mem = np.zeros((s, s))
            self.nmem = np.ones((s, s))  # 定义(spin)与(nspin)
        else:
            sx = (self.diy).shape[0]
            sy = (self.diy).shape[1]
            self.mem = np.zeros((sx, sy))
            self.nmem = np.ones((sx, sy))  # 定义(spin)与(nspin)
            for i in range(0, sx):
                for j in range(0, sy):
                    rand_num = randint(0, 1)
                    if rand_num == 0:
                        self.mem[i,j] = 0
                        self.nmem[i,j] = 1
                    else:
                        self.mem[i, j] = 1
                        self.nmem[i, j] = 0
        self.temp = copy.deepcopy(self.mem)
        self.ntemp = copy.deepcopy(self.nmem)

    def p_bit(self, Vin): # 计算p_bit概率,返回0，1, 此处零点：1.4V
        p = uniform(0, 1)
        if Vin < 0.6:
            probility = 0
        elif Vin > 2.9:
            probility = 1
        else:
            probility = (0.005824121226163 * (Vin ** 5) + (-0.018709074964340) * (Vin ** 4) + (-0.145063118996928) * (Vin ** 3)
                    + 0.629750529077779 * (Vin ** 2) + (-0.216817438963671) * Vin + 0.007384891364228)
        Vout = (p<probility)
        return Vout

    def PRESETspin(self, x, y): # 预置SPIN至HRS
        self.mem[x, y] = 0

    def SET(self, x, y, value):
        if self.mem[x, y] == 0 and value == 1:
            self.mem[x, y] = 1

    def PRESETnspin(self, x, y):  # 预置NSPIN至LRS
        self.nmem[x, y] = 1

    def INV(self, x, y): # 反相操作
        if self.nmem[x, y] == 1 and self.mem[x, y] == 1:
            self.nmem[x, y] = 0

    def TRNG(self, bits_num=5):
        result = 0
        for i in range(0, bits_num):
            result = result + self.p_bit(1.4) * (2 ** (i))
        return result

    def LUT(self, MAC, bits_num=5, range=[-48,48]):
        scale = (range[1]-range[0])/(2 ** bits_num)
        midpoint = 2 ** (bits_num-1)
        result = round(MAC/scale + midpoint)
        return result

    def my_pbit(self, Vin):
        return func(p_reco=p_reco, step_reco=step_reco, x=Vin)

    def single_step(self, x, y, mode):
        if mode == 'x' or mode == 'y':
            rx, ry = self.neighbour(x, y)
        else:
            rx, ry = self.neighbourdiy(x, y)
        lenx = len(rx)
        # 读电流操作
        MAC = 0
        for i_0 in range(0, lenx):
            MAC = (MAC + (self.edge_query(x, y, rx[i_0], ry[i_0], mode)
                          * (self.temp[rx[i_0],ry[i_0]] - self.ntemp[rx[i_0],ry[i_0]])))
        # 跨阻放大器 ： V = -Rf * I
        # V = -self.Rf * I
        # 直流偏置 V = 1.8 + V
        # V = 1.8 + V
        # p-bit
        # p = random()
        rand_num = self.TRNG(6)
        mre = self.LUT(MAC,6)

        self.PRESETspin(x, y)
        if mre < rand_num:
            self.SET(x, y, 1)
        else:
            self.SET(x, y, 0)
        self.PRESETnspin(x, y)
        self.INV(x, y)

    def single_step_pbit(self, x, y, mode):
        if mode == 'x' or mode == 'y':
            rx, ry = self.neighbour(x, y)
        else:
            rx, ry = self.neighbourdiy(x, y)
        lenx = len(rx)
        # 读电流操作
        MAC = 0
        for i_0 in range(0, lenx):
            MAC = (MAC + (self.edge_query(x, y, rx[i_0], ry[i_0], mode)
                          * (self.temp[rx[i_0],ry[i_0]] - self.ntemp[rx[i_0],ry[i_0]]))* (self.Vread) * (1-1/self.onoffratio))
        # 跨阻放大器 ： V = -Rf * I
        V = -self.Rf * MAC
        # 偏置 V = 3.42 + V
        V = 3.42 + V
        # p-bit
        # p = random()
        p = random()
        self.PRESETspin(x, y)
        if p < self.my_pbit(V):
            self.SET(x, y, 1)
        else:
            self.SET(x, y, 0)
        self.PRESETnspin(x, y)
        self.INV(x, y)

    def neighbourdiy(self, x, y, fields=3):
        sx = (self.diy).shape[0]
        sy = (self.diy).shape[1]
        lx = []
        ly = []
        for i0 in range(-1*fields, fields+1):
            for j0 in range(-1*fields, fields+1):
                lx.append(i0)
                ly.append(j0)
        # lx = [-1,-1,-1,0,0,0,1,1,1]
        # ly = [-1,0,1,-1,0,1,-1,0,1]
        rx = []
        ry = []
        for i in range(0, len(lx)):
            newx = x + lx[i]
            newy = y + ly[i]
            if newx >= sx:
                newx = newx - sx
            elif newx < 0:
                newx = sx + newx
            if newy >= sy:
                newy = newy - sy
            elif newy < 0:
                newy = sy + newy
            rx.append(newx)
            ry.append(newy)
        return rx, ry

    def simulatediy(self, max_step=100):
        sx = (self.diy).shape[0]
        sy = (self.diy).shape[1]
        plt.ion()
        E = self.fullE(self.mem, 'diy')
        E_reco = [E]
        # plt.imshow(self.mem)
        # plt.pause(0.1)
        # plt.clf()
        step_reco = [0]
        plt.imshow(self.mem)
        # plt.show()
        plt.pause(1)
        plt.savefig("demo_" + str(0) + ".jpg")
        # plt.clf()
        plt.close()
        for count in range(1, max_step):
            # 一次迭代
            for i in range(0, sx):
                for j in range(0, sy):
                    self.single_step_pbit(i, j, 'diy')
            # plt.imshow(self.mem)
            self.temp = copy.deepcopy(self.mem)
            self.ntemp = copy.deepcopy(self.nmem)
            # plt.imshow(self.mem)
            # plt.pause(0.1)
            # plt.clf()
            E_reco.append(self.fullE(self.mem, 'diy'))
            step_reco.append(count)
            if count % (max_step/10) == 0:
                E_reco.append(self.fullE(self.mem, 'diy'))
                print(E_reco)

                # step_reco.append(count)
                plt.imshow(self.mem)
                plt.pause(1)
                plt.savefig("demonew_"+str(count)+".jpg")
                plt.close()

        # plt.imsave('simu1.jpg', 255*(self.mem))
        numpy.savetxt('step_reco_new.txt', step_reco)
        numpy.savetxt('E_reco_new.txt', E_reco)
        plt.plot(step_reco, E_reco, linewidth =2.0)
        font1 = {'family': 'Arial',
                 'weight': 'normal',
                 'size': 18,
                 }
        plt.xlabel("Cycles", font1)
        plt.ylabel("Energy", font1)
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Arial') for label in labels]

        # plt.ion()
        # plt.imshow(self.mem)
        # fig = plt.figure()
        # plt.pause(20)
        # plt.clf()
        # plt.plot(step_reco, E_reco)
        # plt.show()
        # plt.pause(20)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread(r"demo1.jpg",0)
    # # img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    sp = img.shape
    cv2.imwrite("demo.jpg", img)
    eg = np.zeros((sp[0],sp[1]))
    for i in range(0, sp[0]):
        for j in range(0, sp[1]):
            if img[i, j] == 255:
                eg[i, j] = 1
    # print(img[0, 0])
    # print(sp)
    # cv2.imshow("title", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # a = max_cut()
    # a.simulate('y')
    b = mem_maxcut_sim()
    b.diy_set(eg)
    # b.diy_set(np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]])) #"X"
    # # b.diy_set(np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])) #"Y"
    b.defineproblem('diy')
    b.simulatediy(max_step=100)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
