# -*- coding: utf-8 -*-
"""
Bloch Equation Simulation

"""

import numpy as np
from ..signal.util import vec


# Spolier matrix (only keep z component)
ASP = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])


def trans_relax(t, T2=1.0):
    """
    t and T2 in second.
    return the transverse relaxation matrix A.

    When on-resonance, in rotation coordinates,
        Mx(t) = Mx(0) * exp(-t / T2)
        My(t) = My(0) * exp(-t / T2)
    Then,
        M1 = A * M0

    """
    return np.diag([np.exp(-t / T2), np.exp(-t / T2), 1])


def longi_relax(t, T1=1.0):
    """
    t and T2 in second.
    return the longitudinal relaxation matrix A and B.

    Mathematically, we write
        Mz(t) = M0 + [Mz(0) - M0] * exp(-t / T1)
    Then,
        M1 = A * M0 + B

    """
    A = np.diag([1, 1, np.exp(-t / T1)])
    B = vec([0, 0, 1 - np.exp(-t / T1)], column=True)
    return A, B


def xrot(phi=0):
    """
    Rotation about the x axis.

    """
    Rx = np.eye(3)
    Rx[1,] = [0, np.cos(phi), -np.sin(phi)]
    Rx[2,] = [0, np.sin(phi), np.cos(phi)]
    return Rx


def yrot(phi=0):
    """
    Rotation about the y axis

    """
    Ry = np.eye(3)
    Ry[0,] = [np.cos(phi), 0, np.sin(phi)]
    Ry[2,] = [-np.sin(phi), 0, np.cos(phi)]
    return Ry


def zrot(phi=0):
    """
    Rotation about the z axis.

    """
    Rz = np.eye(3)
    Rz[0,] = [np.cos(phi), -np.sin(phi), 0]
    Rz[1,] = [np.sin(phi), np.cos(phi), 0]
    return Rz


def throt(phi=0, theta=0):
    return zrot(theta) @ xrot(phi) @ zrot(-theta)


def free_precess(t, T1, T2, df=0.0):
    """
    Free precession (in rotation coordinates,) over a
    time inverval `t`, given relaxation times `T1` and
    `T2` and off-resonance frequency `df`. Times in
    second, frequency in Hz.

    """
    phi = 2 * np.pi * df * t  # resonant precession, radians
    E1 = np.exp(-t / T1)
    E2 = np.exp(-t / T2)
    Afp = np.array([[E2, 0, 0], [0, E2, 0], [0, 0, E1]]) @ zrot(phi)
    Bfp = vec([0, 0, 1 - E1], column=True)

    return Afp, Bfp


def ss(flip, T1, T2, TR, df=0.0):
    """
    Calculate the magnetization at steady state for
    repeated excitation given T1, T2, TR in second.
    `df` is the resonant frequency in Hz.
    `flip` is in radians.

    """
    Rflip = yrot(flip)
    [Atr, Btr] = free_precess(TR, T1, T2, df)

    '''
    Calculate steady state
    M1 =  Atr * Rflip * M + Btr

    But M1 = M in steady state, so
        M =  Atr * Rflip * M + Btr
        (I - Atr * Rflip) * M = Btr
    '''
    Mss = np.linalg.pinv(np.eye(3) - Atr @ Rflip) @ Btr
    return Mss


def ss_signal(flip, T1, T2, TE, TR, df=0.0):
    """
    Calculate the steady state signal at TE for repeated
    excitations given T1,T2,TR,TE in second.
    `df` is the resonant frequency in Hz.
    `flip` is in radians.

    """
    Rflip = yrot(flip)
    [Atr, Btr] = free_precess(TR - TE, T1, T2, df)
    [Ate, Bte] = free_precess(TE, T1, T2, df)

    '''
    Let M1 be the magnetization just before the tip.
        M2 be just after the tip.
        M3 be at TE.
    
    then
        M2 = Rflip * M1
        M3 = Ate * M2 + Bte
        M1 = Atr * M3 + Btr
    
    Solve for M3...
        M3 = Ate*Rflip*Atr*M3 + (Ate*Rflip*Btr+Bte)
    '''
    Mte = np.linalg.pinv(np.eye(3) - Ate @ Rflip @ Atr) \
          @ (Ate @ Rflip @ Btr + Bte)

    return Mte


def sr(flip, T1, T2, TR, df=0.0):
    """
    Saturation reconvery.

    Calculate the magnetization at steady state for
    repeated excitation given T1, T2, TR in second.
    Force the transverse magnetization to zero
    before each excitation.
    `df` is the resonant frequency in Hz.
    `flip` is in radians.

    """
    Rflip = yrot(flip)
    [Atr, Btr] = free_precess(TR, T1, T2, df)

    '''
    Force transverse magnetization to 0 before excitation
    '''
    Atr = ASP @ Atr

    '''
    Calculate steady state
    M1 =  Atr * Rflip * M + Btr

    But M1 = M in steady state, so
        M =  Atr * Rflip * M + Btr
        (I - Atr * Rflip) * M = Btr
    '''
    Mss = np.linalg.pinv(np.eye(3) - Atr @ Rflip) @ Btr
    return Mss


def sr_signal(flip, T1, T2, TE, TR, df=0.0):
    """
    Calculate the steady state signal at TE for repeated
    excitations given T1,T2,TR,TE in second. Force the
    transverse magnetization to zero before each excitation.
    `df` is the resonant frequency in Hz.
    `flip` is in radians.

    """
    Rflip = yrot(flip)
    [Atr, Btr] = free_precess(TR - TE, T1, T2, df)
    [Ate, Bte] = free_precess(TE, T1, T2, df)

    '''
    Force transverse magnetization to 0 before excitation
    '''
    Atr = np.array([[0,0,0],[0,0,0],[0,0,1]]) @ Atr

    '''
    Let M1 be the magnetization just before the tip.
        M2 be just after the tip.
        M3 be at TE.
    
    then
        M2 = Rflip * M1
        M3 = Ate * M2 + Bte
        M1 = Atr * M3 + Btr
    
    Slove for M3...
        M3 = Ate*Rflip*Atr*M3 + (Ate*Rflip*Btr+Bte)
    '''
    Mte = np.linalg.pinv(np.eye(3) - Ate @ Rflip @ Atr) \
          @ (Ate @ Rflip @ Btr + Bte)
    return Mte


def se(T1, T2, TE, TR, df=0.0, dt=0.001, M0=None):
    """
    Spin echo at first TR.
    T1, T2, TE, TR, dt in second.
    df of off-resonance frequence in Hz.

    """
    N1 = round(TE / 2 / dt)
    N2 = round((TR - TE / 2) / dt)

    ''' Get the propagation matrix '''
    [A1, B1] = free_precess(dt, T1, T2, df)

    ''' Simulate the decay '''
    M = np.zeros([3, N1 + N2])

    if M0 is None:
        M[:, 0] = [0, 0, 1]  # starting magnetization
    else:
        M[:, 0] = vec(M0)

    Rflip = yrot(np.pi / 2)
    Rrefoc = xrot(np.pi)

    # Excitation
    M[:, 1:2] = A1 @ Rflip @ M[:, 0:1] + B1
    for k in range(2, N1):
        M[:, k:k + 1] = A1 @ M[:, k - 1:k] + B1

    # Refocus
    M[:, N1:N1 + 1] = A1 @ Rrefoc @ M[:, N1 - 1:N1] + B1
    for k in range(N1 + 1, N1 + N2):
        M[:, k:k + 1] = A1 @ M[:, k - 1:k] + B1

    return M


def sess(T1, T2, TE, TR, df=0.0):
    """
    Calculate the magnetization at steady state for
    a spin-echo sequence, given T1, T2, TR in second.
    Force the transverse magnetization to zero
    before each excitation.
    `df` is the resonant frequency in Hz.

    """
    Rflip = yrot(np.pi / 2)  # Rotation from excitation pulse (90)
    Rrefoc = xrot(np.pi)  # Rotation from refocusing pulse (180)

    [Atr, Btr] = free_precess(TR - TE / 2, T1, T2, df)
    [Ate, Bte] = free_precess(TE / 2, T1, T2, df)

    '''
    Neglect residual transverse magnetization prior to excitation
    '''
    # Just keep the z component
    Atr = ASP @ Atr

    '''
    Let M1 be the magnetization just before the pi/2 pulse
        M2 be just before the pi pulse
    
    then,
        M2 = Ate * Rflip * M1 + Bte
        M1 = Atr * Rrefoc * M2 + Btr
    
    Solve for M1...
        M1 = Atr * Rrefoc * (Ate * Rflip * M1 + Bte) + Btr
        (I - Atr * Rrefoc * Ate * Rflip) * M1 = Atr * Rrefoc * Bte + Btr
    '''
    left = np.eye(3) - Atr @ Rrefoc @ Ate @ Rflip
    right = Atr @ Rrefoc @ Bte + Btr
    Mss = np.linalg.pinv(left) @ right

    ''' The steady state magnetization at TE '''
    Mte = Ate @ Rflip @ Mss + Bte
    Mte = Ate @ Rrefoc @ Mte + Bte

    return Mss, Mte


def sess_signal(T1, T2, TE, TR, df=0.0):
    """
    Direct calculate the steady state magnetization at TE
    for a spin-echo sequence, given T1, T2, TR in second.
    Force the transverse magnetization to zero
    before each excitation.
    `df` is the resonant frequency in Hz.

    """
    Rflip = yrot(np.pi / 2)  # Rotation from excitation pulse (90)
    Rrefoc = xrot(np.pi)  # Rotation from refocusing pulse (usually 180)

    [Atr, Btr] = free_precess(TR - TE, T1, T2, df)
    [Ate, Bte] = free_precess(TE / 2, T1, T2, df)

    '''
    Neglect residual transverse magnetization prior to excitation
    '''
    # Just keep the z component
    Atr = ASP @ Atr

    '''
    Let M1 be the magnetization just before the pi/2 pulse
        M2 be just before the pi pulse
        M3 be at TE

    then,
        M2 = Ate * Rflip * M1 + Bte
        M3 = Ate * Rrefoc * M2 + Bte
        M4 = Atr * M3 + Btr
        M1 = M4

    Solve for M3...
        M3 = Ate * Rrefoc * (Ate * Rflip * (Atr * M3 + Btr) + Bte) + Bte
           = Ate * Rrefoc * (Ate * Rflip * Atr * M3 + Ate * Rflip * Btr + Bte) + Bte
        
        (I - Ate * Rrefoc * Ate * Rflip * Atr) * M3 
        = Ate * Rrefoc * (Ate * Rflip * Btr + Bte) + Bte
    '''
    left = np.eye(3) - Ate @ Rrefoc @ Ate @ Rflip @ Atr
    right = Ate @ Rrefoc @ (Ate @ Rflip @ Btr + Bte) + Bte
    Mte = np.linalg.pinv(left) @ right

    return Mte


def fse(T1, T2, TE, TR, df=0.0, dt=0.001, ETL=1, M0=None):
    """
    Fast spin echo at first TR.
    T1, T2, TE, TR, dt in second.
    df of off-resonance frequence in Hz.
    ETL : Echo train length >= 1

    """
    N1 = round(TE / 2 / dt)
    N2 = N1 * 2
    N3 = round((TR - TE / 2 - TE * (ETL - 1)) / dt)

    if not TR > (TE / 2 + TE * (ETL - 1)):
        raise ValueError(f'Given TR {TR}s is not enough to contain {ETL} echoes.')

    ''' Get the propagation matrix '''
    [A1, B1] = free_precess(dt, T1, T2, df)

    ''' Simulate the sequence '''
    M = np.zeros([3, N1 + N2 * ETL + N3])

    if M0 is None:
        M[:, 0] = [0, 0, 1]  # starting magnetization
    else:
        M[:, 0] = vec(M0)

    Rflip = yrot(np.pi / 2)
    Rrefoc = xrot(np.pi)

    # Excitation
    M[:, 1:2] = A1 @ Rflip @ M[:, 0:1] + B1
    for k in range(2, N1):
        M[:, k:k + 1] = A1 @ M[:, k - 1:k] + B1

    # Refocus
    for e in range(ETL):
        start = N1 + N2 * e
        M[:, start:start + 1] = A1 @ Rrefoc @ M[:, start - 1:start] + B1

        for k in range(start + 1, start + N2):
            M[:, k:k + 1] = A1 @ M[:, k - 1:k] + B1

    # From last TE to the end
    start = N1 + N2 * ETL
    for k in range(start, start + N3):
        M[:, k:k + 1] = A1 @ M[:, k - 1:k] + B1

    return M


def fse_signal(T1, T2, TE, TR, df=0.0, ETL=1):
    """
    Calculate the steady state magnetization at TE for a multi-echo spin-echo
    sequence, given T1, T2, TE, TR, dt in second.
    Force the transverse magnetization to zero before each excitation.
    df of off-resonance frequence in Hz.
    ETL : Echo train length >= 1

    """
    Rflip = yrot(np.pi / 2)
    Rrefoc = xrot(np.pi)

    [Atr, Btr] = free_precess(TR - ETL * TE, T1, T2, df)
    [Ate, Bte] = free_precess(TE / 2, T1, T2, df)

    ''' Neglect residual transverse magnetization prior to excitation '''
    Atr = ASP @ Atr

    '''
    Since ETL varies, let's keep a 'running' A and B.
    We'll calculate the steady-state signal just after the tip, Rflip.
    '''

    ''' Initial '''
    A = Rflip
    B = np.array([0, 0, 0]).reshape(3, 1)

    ''' For each echo, we 'propagate' A and B
    From the center of last echo to the center of next echo
    '''
    for k in range(ETL):
        A = Ate @ Rrefoc @ Ate @ A
        B = Ate @ Rrefoc @ (Ate @ B + Bte) + Bte

    ''' Propagate A and B through to just before flip,
    and force the transverse magnetization to zero,
    and calculate steady-state
    '''
    A = ASP @ Atr @ A
    B = Atr @ B + Btr

    # Steady state is right before pi/2 pulse
    Mss = np.linalg.pinv(np.eye(3) - A) @ B
    M = Rflip @ Mss

    ''' Calculate signal on each echo '''
    Mte = np.zeros([3, ETL])
    for k in range(ETL):
        M = Ate @ Rrefoc @ (Ate @ M + Bte) + Bte
        Mte[:, k:k+1] = M

    return Mss, Mte


def ge(flip, T1, T2, TE, TR, df, phi):
    """
    Gradient echo at first TR.
    T1, T2, TE, TR, dt in second.
    df of off-resonance frequence in Hz.
    phi is the phase twist at the end of the sequence in radians.

    """
    pass


def gess(flip, T1, T2, TE, TR, df, phi):
    """
    Calculate the magnetization at steady state for
    a gradient-spoiled sequence, given T1, T2, TR in second.
    `df` is the resonant frequency in Hz.
    phi is the phase twist at the end of the sequence in radians.

    """
    Rflip = yrot(flip)
    [Atr, Btr] = free_precess(TR, T1, T2, df)

    '''
    To add the gradient spoiler twist, we just multiply Atr by
    zrot(phi). (Dephasing)
    '''
    Atr = zrot(phi) @ Atr

    '''
    Let M1 be the magnetization just before the tip, then
        M1 = Atr @ Rflip @ M1 + Btr
    Solve for M1...
    '''
    Mss = np.linalg.pinv(np.eye(3) - Atr @ Rflip) @ Btr
    return Mss


def gess_signal(flip, T1, T2, TE, TR, df, phi, N=1):
    """
    Calculate the average steady state magnetization in one voxel
    at TE for a gradient-spoiled sequence, given T1, T2, TR in second.
    `df` is the resonant frequency in Hz.
    phi is the phase twist across the voxel in radians.
    N is the number of spins in one voxel.

    """
    M = np.zeros([3, N])
    if N == 1:
        p = [phi, ]
    else:
        p = (np.array(range(N)) / N - 0.5) * phi

    for k in range(N):
        Mte = _gess_signal(flip, T1, T2, TE, TR, df, p[k])
        M[:, k:k + 1] = Mte

    Mte_avg = np.mean(M, axis=1)
    return Mte_avg


def _gess_signal(flip, T1, T2, TE, TR, df, phi):
    """
    Calculate the steady state magnetization at TE for
    a gradient-spoiled sequence, given T1, T2, TR in second.
    `df` is the resonant frequency in Hz.
    phi is the phase twist at the end of the sequence in radians.

    """
    Rflip = yrot(flip)
    [Atr, Btr] = free_precess(TR - TE, T1, T2, df)
    [Ate, Bte] = free_precess(TE, T1, T2, df)

    '''
    To add the gradient spoiler twist, we just multiply Atr by
    zrot(phi). (Dephasing)
    '''
    Atr = zrot(phi) @ Atr

    '''
    Force the transverse magnetization to zero before excitation
    '''
    # Atr = ASP @ Atr

    '''
    Let M1 be the magnetization just before the tip.
        M2 be at TE.
    then,
        M2 = Ate @ (Rflip @ M1) + Bte
        M1 = Atr @ M2 + Btr
        
    Solve for M2...
        M2 = Ate @ Rflip @ (Atr @ M2 + Btr) + Bte
        (I - Ate @ Rflip @ Atr) @ M2 = Ate @ Rflip @ Btr + Bte
    '''
    left = np.eye(3) - Ate @ Rflip @ Atr
    right = Ate @ Rflip @ Btr + Bte
    Mte = np.linalg.pinv(left) @ right
    return Mte
