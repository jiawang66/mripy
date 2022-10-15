# -*- coding: utf-8 -*-
"""
Test Bloch Simulation

References : <http://mrsrl.stanford.edu/~brian/bloch/>
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
sys.path.append('../../../mripy')
from mripy.mri import bloch


def test_freeprecess():
    print('\n****** Test free precession ******')

    dt = 0.001  # 1ms delta-time
    T = 1       # total dulation
    N = int(np.ceil(T / dt) + 1)
    df = 10     # Hz off-resonance
    T1 = 0.6    # second
    T2 = 0.1    # second

    ''' Get the propagation matrix '''
    [A, B] = bloch.free_precess(dt, T1, T2, df)

    ''' Simulate the decay '''
    M = np.zeros([3, N])
    M[:, 0] = [1, 0, 0]

    for k in range(1, N):
        M[:, k:k + 1] = A @ M[:, k - 1:k] + B

    ''' Plot the results '''
    time = np.array(range(N)) * dt
    plt.plot(time, M[0,])
    plt.plot(time, M[1,])
    plt.plot(time, M[2,])
    plt.legend(['Mx', 'My', 'Mz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetization')
    plt.title('Free precession after pi/2 excitation')
    plt.axis([np.min(time), np.max(time), -1, 1])
    plt.grid()
    plt.show()


def test_B_1a():
    print('\n****** Magnetization at 1ms after the first excitation ******')

    df = 0  # Hz
    T1 = 0.6  # second
    T2 = 0.1  # second
    TE = 0.001  # second
    TR = 0.5  # second
    flip = np.pi / 3  # raidans

    Rflip = bloch.yrot(flip)
    [Ate, Bte] = bloch.free_precess(TE, T1, T2, df)
    [Atr, Btr] = bloch.free_precess(TR, T1, T2, df)

    ''' Magnetization at TE after the first excitation '''
    M = np.array([0, 0, 1]).reshape(3, 1)
    M = Rflip @ M
    M = Ate @ M + Bte
    print(M.transpose())

    ''' Magnetization at TE after the second excitation '''
    print('\n****** Magnetization at 1ms after the second excitation ******')
    M = np.array([0, 0, 1]).reshape(3, 1)
    M = Rflip @ M
    M = Atr @ M + Btr
    M = Rflip @ M
    M = Ate @ M + Bte
    print(M.transpose())


def test_B_1c():
    print('\n****** Magnetization varies over the first 10 excitations ******')

    df = 0  # Hz
    T1 = 0.6  # second
    T2 = 0.1  # second
    dt = 0.001
    TR = 0.5  # second
    flip = np.pi / 3  # raidans
    Ntr = round(TR / dt)
    Nex = 10  # number of excitations

    M0 = np.array([0, 0, 1]).reshape(3, 1)
    Rflip = bloch.yrot(flip)
    [A1, B1] = bloch.free_precess(dt, T1, T2, df)

    M = np.zeros([3, Nex * Ntr + 1])
    M[:, 0] = M0.reshape(-1)

    Mcount = 0
    for n in range(Nex):
        Mcount += 1
        M[:, Mcount:Mcount+1] = Rflip @ M[:, Mcount-1:Mcount]

        for k in range(Ntr - 1):
            Mcount += 1
            M[:, Mcount:Mcount+1] = A1 @ M[:,Mcount-1:Mcount] + B1

    ''' Plot the results '''
    time = np.array(range(Mcount + 1)) * dt
    plt.plot(time, M[0,])
    plt.plot(time, M[1,])
    plt.plot(time, M[2,])
    plt.legend(['Mx', 'My', 'Mz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetization')
    plt.title('Magnetization varies over the first 10 excitations')
    plt.axis([np.min(time)-0.1, np.max(time), -1, 1])
    plt.grid()
    plt.show()

    ''' B-1d
    Calculate steady state
    '''
    print('\n****** Magnetization at steady state ******')
    M = bloch.ss(flip, T1, T2, TR, df)
    print(M.transpose())


def test_B_1e():
    print('\n****** The steady-state magnetization at the echo time ******')
    df = 0  # Hz
    T1 = 0.6  # second
    T2 = 0.1  # second
    TE = 0.001  # second
    TR = 0.5  # second
    flip = np.pi / 3  # raidans

    Rflip = bloch.yrot(flip)
    [Ate, Bte] = bloch.free_precess(TE, T1, T2, df)

    ''' Calculation using B-1d first '''
    Mss = bloch.ss(flip, T1, T2, TR, df)
    Mte_1 = Ate @ Rflip @ Mss + Bte
    print('Calculate Mss first')
    print('Mss', Mss.transpose())
    print('Mte_1', Mte_1.transpose())

    ''' Direct calculation at TE '''
    Mte_2 = bloch.ss_signal(flip, T1, T2, TE, TR, df)
    print('\nDirect calculation at TE')
    print('Mte_2', Mte_2.transpose())

    ''' B-1g '''
    print('\n****** The steady-state magnetization at the echo time ******')
    print('****** Force the transverse magnetization to zero before each excitation ******')
    print('Calculate Mss first')
    Mss = bloch.sr(flip, T1, T2, TR, flip)
    Mte_3 = Ate @ Rflip @ Mss + Bte
    print('Mss', Mss.transpose())
    print('Mte_4', Mte_3.transpose())

    print('\nDirect calculation at TE')
    Mte_4 = bloch.sr_signal(flip, T1, T2, TE, TR, df)
    print('Mte_3', Mte_4.transpose())


def test_B_1f():
    pass
    '''
    Solution to B-1f.
    -----------------
    
    Let M0 be the magnetization just before the excitation.
    Let M1 be the magnetization just afterward.
    
    Tips are about y.
    
    M1z = M0z*cos(flip) - M0x*sin(flip)
    M0z = M1z*E1 + (1-E1)
        where E1 = exp(-TR/T1)
    
    M0x = 0, since we neglect residual transverse magnetization.
    
    Combine these equations:
    
        M0z = [M0z*cos(flip) - 0]*E1 + (1-E1)
        M0z [1-E1*cos(flip)] = 1-E1
        M0z = [1-E1] / [1-E1*cos(flip)]

    ************ Notes ************
    
    1.  It wouldn't be that difficult to keep the M0x term around,
        and not neglect the transverse magnetization.
    
    2.  Multiplication by [0 0 0;0 0 0;0 0 1] just before the excitation
        in B-1d would "force" the transverse magnetization to be zero
        in the matrix calculation, so you should get the same answer
        in B-1d as you just got here in B-1f.
    '''


def test_B_2a():
    print('\n****** SE sequence ******')

    df = 10  # Hz
    T1 = 0.6  # second
    T2 = 0.1  # second
    TE = 0.05  # second
    TR = 0.5  # second
    dt = 0.001 # second

    ''' Simulate spin echo '''
    M = bloch.se(T1, T2, TE, TR, df, dt)

    ''' Plot the results '''
    time = np.array(range(M.shape[1])) * dt
    plt.plot(time, M[0,])
    plt.plot(time, M[1,])
    plt.plot(time, M[2,])
    plt.legend(['Mx', 'My', 'Mz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetization')
    plt.title('Magnetization of SE sequence at first TR')
    plt.axis([np.min(time) - 0.02, np.max(time), -1, 1])
    plt.grid()
    plt.show()

    print('Note that the magnetization has a spin-echo at 50 ms '
          '-- it points along x at this point.')

    Msig = M[0,] + 1j * M[1,]
    plt.plot(time, np.abs(Msig))
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of Magnetization')
    plt.axis([np.min(time) - 0.02, np.max(time), -1, 1])
    plt.grid()
    plt.show()


def test_B_2b():
    print('\n****** SE sequence with different df ******')

    Nf = 20  # Number of frequence
    df = 50 * (np.random.random(Nf) - 0.5)  # Hz
    T1 = 0.6  # second
    T2 = 0.1  # second
    TE = 0.05  # second
    TR = 0.5  # second
    dt = 0.001  # second

    N1 = round(TE / 2 / dt)
    N2 = round((TR - TE / 2) / dt)

    ''' Simulate signal '''
    Msig = np.zeros([Nf, N1 + N2], dtype=np.complex128)
    for f in range(Nf):
        M = bloch.se(T1, T2, TE, TR, df[f], dt)

        # Keep the transverse component
        Msig[f,] = M[0,] + 1j * M[1,]

    ''' Display the results '''
    time = np.array(range(N1 + N2)) * dt
    plt.subplot(2, 1, 1)
    plt.plot(time, np.abs(Msig.transpose()))
    plt.ylabel('Magnitude')
    plt.axis([np.min(time) - 0.02, np.max(time), -1, 1])
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, np.angle(Msig.transpose()))
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (radians)')
    plt.axis([np.min(time) - 0.02, np.max(time), -np.pi, np.pi])
    plt.grid()
    plt.show()

    plt.plot(time, np.abs(np.mean(Msig, axis=0)))
    plt.xlabel('Time (s)')
    plt.ylabel('Net Magnitude')
    plt.axis([np.min(time) - 0.02, np.max(time), -1, 1])
    plt.grid()
    plt.show()

    print('This is a nice simulation of what happens when you have a \n'
          'signal that is the sum of many spins, which is what you \n'
          'actually have. The power of Bloch simulations should be \n'
          'starting to show.')

    ''' Steady state '''
    print('\n****** Steady state of spin-echo sequence ******')
    df = 0
    T1 = 0.6
    T2 = 0.1
    TE = 0.05
    TR = 1

    [Mss, Mte_1] = bloch.sess(T1, T2, TE, TR, df)
    print('Calculate Mss first')
    print('Mss', Mss.transpose())
    print('Mte_1', Mte_1.transpose())

    Mte_2 = bloch.sess_signal(T1, T2, TE, TR, df)
    print('\nDirect calculation at TE')
    print('Mte_2', Mte_2.transpose())


def test_B_2d():
    print('\n****** Fast Spin Echo ******')

    df = 10  # Hz
    T1 = 0.6  # second
    T2 = 0.1  # second
    TE = 0.05  # second
    TR = 1  # second
    ETL = 8  # echo train length
    dt = 0.001  # second

    ''' Simulate signal '''
    M = bloch.fse(T1, T2, TE, TR, df, dt, ETL)
    Msig = M[0,] + 1j * M[1,]

    ''' Display the signal '''
    time = np.array(range(M.shape[1])) * dt
    plt.plot(time, M[0,])
    plt.plot(time, M[1,])
    plt.plot(time, M[2,])
    plt.legend(['Mx', 'My', 'Mz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetization')
    plt.title(f'Magnetization of FSE sequence at first TR (ETL={ETL})')
    plt.axis([np.min(time) - 0.02, np.max(time), -1, 1])
    plt.grid()
    plt.show()

    plt.plot(time, np.abs(Msig))
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.title(f'FSE at first TR with ETL={ETL}')
    plt.axis([np.min(time) - 0.02, np.max(time), -1, 1])
    plt.grid()
    plt.show()

    ''' Steady state '''
    print('\n****** Steady state of FSE sequence ******')
    [Mss, Mte] = bloch.fse_signal(T1, T2, TE, TR, df, ETL)
    print('Mss', Mss.transpose())
    print('Mte_x', Mte[0,])
    print('Notice that the amplitude of the first echo is smaller '
          'in B-2d than B-2c (why?). '
          'If you set ETL to 1, they should be the same.')


def test_B_3a():
    print('\n****** Gradient spoiled ******')

    flip = np.pi / 3  # radians
    T1 = 0.6  # second
    T2 = 0.1  # second
    TE = 0.002  # second
    TR = 0.01  # second
    df = 0  # Hz
    phi = np.pi / 2  # radians
    dt = 0.001  # second

    Rflip = bloch.yrot(flip)
    [Ate, Bte] = bloch.free_precess(TE, T1, T2, df)

    ''' Steady state '''
    print('Steadt state')
    Mss = bloch.gess(flip, T1, T2, TE, TR, df, phi)
    print('Mss', Mss.transpose())

    ''' Steady state at TE '''
    print('\nSteady state at TE')
    Mte_1 = Ate @ Rflip @ Mss + Bte
    print('Calculate Mss first')
    print('Mte_1', Mte_1.transpose())

    Mte_2 = bloch.gess_signal(flip, T1, T2, TE, TR, df, phi)
    print('Direct calculation')
    print('Mte_2', Mte_2.transpose())

    ''' Average magnetization in one voxel '''
    print('\nAverage steady state at TE of one voxel')
    N = 200
    phi = 4 * np.pi
    Mte = bloch.gess_signal(flip, T1, T2, TE, TR, df, phi, N=N)
    print('Mte', Mte.transpose())


def test_B_4a():
    print('\n****** Steady-state Free-Precession ******')

    Nte = 5
    Nf = 200
    flip = np.pi / 3  # radians
    T1 = 0.6  # second
    T2 = 0.1  # second
    TE = np.linspace(0, 10, Nte) * 0.001  # second
    TR = 0.01  # second
    df = np.linspace(-100, 100, Nf)  # Hz

    sig = np.zeros([Nf, Nte], dtype=np.complex128)

    for n in range(Nte):
        for k in range(Nf):
            Mte = bloch.ss_signal(flip, T1, T2, TE[n], TR, df[k]).reshape(-1)
            sig[k, n] = Mte[0] + 1j * Mte[1]

    ''' Plot the results '''
    t = slice(0, Nte)
    plt.subplot(2, 1, 1)
    plt.plot(df, np.abs(sig[:, t]))
    plt.ylabel('Magnitude')
    plt.grid()
    plt.axis([np.min(df), np.max(df), 0, np.max(np.abs(sig))])
    plt.legend(['TE=0', 'TE=2.5', 'TE=5', 'TE=7.5', 'TE=10'])

    plt.subplot(2, 1, 2)
    plt.plot(df, np.angle(sig[:, t]))
    plt.xlabel('Frequency (HZ)')
    plt.ylabel('Phase (radians)')
    plt.axis([np.min(df), np.max(df), -np.pi, np.pi])
    plt.grid()
    plt.show()


def test_B_4b():
    print('\n****** SS ******')


def main():
    # test_freeprecess()
    #
    # ''' Saturation recovery '''
    # test_B_1a()
    # test_B_1c()
    # test_B_1e()
    #
    # ''' Spin-Echo sequence '''
    # test_B_2a()
    # test_B_2b()
    # test_B_2d()
    #
    # ''' Gradient-spoiled sequence '''
    # test_B_3a()

    ''' Steady-state free-precession '''
    test_B_4a()


if __name__ == '__main__':
    main()
