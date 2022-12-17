import cv2
import numpy as np
import math
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
import math
import random
from scipy import interpolate


class quantum:
    def __init__(self):
        #self.acc = IBMQ.save_account('c530ca480a7ae236618ee507628af36c57d2c57cfb3416b99c60054bf6266ab37adecabb40bc92bd32a5609713d17b93736cf6b961ac523eac79fe1c02a6e1ed')
        self.acc = IBMQ.load_account()
        print(self.acc)

    def qft_dagger(self, qc, n):
        # swap
        for qb in range(n // 2):
            qc.swap(qb, n - qb - 1)
        for j in range(n):
            for m in range(j):
                qc.cp(-math.pi / float(2 ** (j - m)), m, j)
            qc.h(j)

    def linear(self, numShots):
        nbits = 1
        circuit = QuantumCircuit(nbits, nbits)
        circuit.h(0)
        circuit.measure([0], [0])
        #circuit.draw(output='mpl')

        sim = Aer.get_backend('qasm_simulator')
        sim_result = execute(circuit, backend=sim, shots=numShots, memory=True)
        data = sim_result.result().get_memory()
        return data

    def linearHardware(self, numShots):
        provider = IBMQ.get_provider('ibm-q')
        qcomp = provider.get_backend('ibmq_quito')
        nbits = 1
        circuit = QuantumCircuit(nbits, nbits)
        circuit.h(0)
        circuit.measure([0], [0])
        # circuit.draw(output='mpl')

        #sim = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend=qcomp, shots=numShots, memory=True)
        job_monitor(job)
        data = job.result().get_memory()
        return data

    def linearModify(self, channel, passes, hw):
        acc = 255/passes
        shots = (channel.dimx * 2)*passes
        if hw == True:
            mem = self.linearHardware(shots)
        else:
            mem = self.linear(shots)

        i = 0
        for x in range(passes):
            x = 0
            y = 0
            while (x < channel.dimx) and (y < channel.dimy):
                channel.addBit(x, y, acc)
                if mem[i] == '0':
                    x += 1
                else:
                    y += 1
                i += 1

    def quantum_phase_est(self, a, numQB, passes, hw):

        qc = QuantumCircuit(numQB, numQB - 1)
        qc.x(numQB - 1)

        # apply hadamard gate to each counting bit
        for qb in range(numQB - 1):
            qc.h(qb)

        # apply controlled unitary operation
        reps = 1
        for counter in range(numQB - 1):
            for i in range(reps):
                qc.cp((a), counter, numQB - 1)
            reps *= 2

        qc.barrier()
        self.qft_dagger(qc, numQB - 1)

        qc.barrier()
        for n in range(numQB - 1):
            qc.measure(n, n)
        #qc.draw('mpl')
        #plt.show()

        if hw:

            provider = IBMQ.get_provider('ibm-q')
            sim = provider.get_backend('ibmq_quito')
        else:
            sim = Aer.get_backend('aer_simulator')
        shots = passes
        t_qc = transpile(qc, sim)

        qobj = assemble(t_qc, shots=shots)
        result = sim.run(qobj).result()
        answer = result.get_counts()
        print(answer)
        sum = 0
        for key in answer.keys():
            sum += bin_to_float(key) * answer[key]
            print(bin_to_float(key), answer[key])

        print("answer:", sum / shots)
        #_histogram(answer)
        #plt.show()
        return answer


class channel:
    def __init__(self, sizex, sizey):
        self.data = np.zeros((sizex, sizey))
        self.dimx = sizex
        self.dimy = sizey

    def setBit(self, x, y, val):
        self.data[x][y] = val

    def addBit(self, x, y, val):
        if (self.data[x][y] + val) >= 255:
            self.data[x][y] = 255
        else:
            self.data[x][y] += val

    def save(self, fName):
        np.savetxt(fName, self.data)

    def randomize(self):
        self.data = np.random.randint(50, size=(self.dimx, self.dimy))


class cv2Wrapper:
    def __init__(self):
        self.prepared = False
        self.img = None
        self.w = 0
        self.h = 0

    def combine(self, red, green, blue, w, h):
        self.img = np.dstack((red, green, blue))
        self.w = w
        self.h = h
        self.prepared = True

    def show(self, doBlur, blur):
        if self.prepared:

            newSize = (self.w, self.h)
            gen = np.array(self.img, dtype=np.uint8)
            gen = cv2.resize(gen, newSize, interpolation=cv2.INTER_NEAREST)
            if doBlur:
                gen = cv2.GaussianBlur(gen, (5,5),0)
            self.img = gen

            cv2.imshow('i', gen)
            cv2.waitKey(2000)
            cv2.destroyWindow('i')
        else:
            print("not ready, please load channel data")

    def save(self, fName):
        cv2.imwrite(fName, self.img)


def bin_to_float(bin):
    total = 0
    for i in range(1, len(bin)+1):
        if bin[i - 1] == '1':
            total = total + 2 ** (-1*i)

    return total


def linearVis(dimension, passes, hw):
    n = dimension
    red = channel(n, n)
    green = channel(n, n)
    blue = channel(n, n)

    # red.randomize()
    # green.randomize()
    q = quantum()
    q.linearModify(blue, passes, hw)
    q.linearModify(red, passes, hw)
    q.linearModify(green, passes, hw)

    result = cv2Wrapper()
    result.combine(red.data, green.data, blue.data, 900, 900)
    result.show(0, False)
    result.save("n_"+str(n)+"passes_"+str(passes)+".png")


def plotLagrange(est, channel, dimension, passes, qb):
    x = []
    y = []

    for key in sorted(est.keys()):
        y.append(est[key])
        x.append(bin_to_float(key)*dimension)

    print(x, y)

    f = interpolate.UnivariateSpline(x, y)

    print(f)
    for newX in range(0, dimension):
        newY = f(newX)
        for i in range(0, int(newY)):

        #print(newX, newY)
            if (i < 900) and (newX < 900):
                channel.setBit(newX, i, 255)
    '''
    fig = plt.figure(figsize=(10, 8))
    plt.plot(x_new, f(x_new), 'b', x, y, 'ro')
    plt.title('Lagrange Polynomial')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    '''
def plotHist(est, channel, dimension, passes, qb):
    index = 0
    for key in sorted(est.keys()):
        val = bin_to_float(key)
        print("val", val)
        for y in range(0, math.floor(est[key] * (dimension / passes))):
            width = int(dimension / (2 ** (qb - 1)))
            x = width * index
            for i in range(x, x+width):
                channel.setBit(i, y, 255)
        index += 1


def QFEVis(a, dimension, passes, hw, qb):
    q = quantum()
    n = dimension
    red = channel(n, n)
    green = channel(n, n)
    blue = channel(n, n)

    est = q.quantum_phase_est(a, qb, passes*2, hw)
    plotLagrange(est, blue, dimension, passes, qb)
    est = q.quantum_phase_est(a, qb, passes//2, hw)
    plotLagrange(est, green, dimension, passes, qb)
    est = q.quantum_phase_est(a, qb, passes, hw)
    plotLagrange(est, red, dimension, passes, qb)


    result = cv2Wrapper()
    result.combine(blue.data, green.data, red.data, 900, 900)
    result.show(7, True)
    result.save("n_"+str(n)+"passes_"+str(passes)+".png")

def main():
    #linearVis(900, 255, False)
    a = (2*math.pi)/3
    QFEVis(a, 900, 1048, True, 5)

if __name__ == "__main__":
    main()

