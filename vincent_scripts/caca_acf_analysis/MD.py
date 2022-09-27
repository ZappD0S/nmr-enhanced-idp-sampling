#!/usr/bin/env python
import numpy as np
from statsmodels.tsa.stattools import acf

def normalize(vect):
    x, y, z = vect[0], vect[1], vect[2]
    norm = (x*x + y*y + z*z)**(1/2)
    if norm == 0:
        print("you gave me a null vector...")
        return 0
    return [x/norm, y/norm, z/norm]

def autocorr(x, lags="default"):
    '''using statsmodels.tsa.stattools.acf'''
    if lags == "default":
        return acf(x, nlags=len(x)//4, fft=True)
    elif lags.is_integer():
        return acf(x, nlags=int(lags), fft=True)
    else:
        return acf(x, fft=True)

def rotacf(tvl):
    # tvl = time vector list: v[time][xyz]
    # returns the orientation autocorrelation of the given vector
    T = len(tvl)
    norm_tvl = [normalize(tvl[i]) for i in range(T)]
    xx = np.array([norm_tvl[i][0]*norm_tvl[i][0] for i in range(T)])
    yy = np.array([norm_tvl[i][1]*norm_tvl[i][1] for i in range(T)])
    zz = np.array([norm_tvl[i][2]*norm_tvl[i][2] for i in range(T)])
    xy = np.array([norm_tvl[i][0]*norm_tvl[i][1] for i in range(T)])
    xz = np.array([norm_tvl[i][0]*norm_tvl[i][2] for i in range(T)])
    yz = np.array([norm_tvl[i][1]*norm_tvl[i][2] for i in range(T)])
    XX = autocorr(xx)
    YY = autocorr(yy)
    ZZ = autocorr(zz)
    XY = autocorr(xy)
    XZ = autocorr(xz)
    YZ = autocorr(yz)
    output = [(XX[i] + YY[i] + ZZ[i] + 2*XY[i] + 2*XZ[i] + 2*YZ[i])/9 for i in range(len(XX))]
    return output

'''
############## Relaxation

theta = 22

gamma15N = -2.7126 * 1e7
gamma1H = 2.6752219 * 1e8
rNH = 1.015 * 1e-10
tCSA = -172.0 * 1e-6
mu_0 = 1.2566370614359173 * 1e-6
h = 6.62607004e-34

d = mu_0*h*gamma1H*gamma15N/(8*(np.pi**2)*(rNH**3))
theta = theta*np.pi/180

def P2(x):
    return (3*x*x - 1)/2

def j(omega, amp1, amp2, t1, t2, t3):
    return amp1*t1/(1+((omega**2)*(t1**2))) + amp2*t2/(1+((omega**2)*(t2**2))) + (1- amp1 - amp2)*t3/(1+((omega**2)*(t3**2)))

def jj(omega, amps, taus):
    return sum([amps[i]*taus[i]/(1+((omega**2)*(taus[i]**2))) for i in range(len(amps))])

def R1(x, amps, taus):
    #def J(omega): return j(omega, amp1, amp2, t1, t2, t3)
    def J(omega): return jj(omega, amps, taus)
    field = x * 1e6 / (gamma1H/(2*np.pi))
    omegaN = gamma15N * field
    omegaH = gamma1H * field
    omega_sum = omegaH + omegaN
    omega_diff = omegaH - omegaN
    c = tCSA * omegaN
    return ((d**2)/10)*(6*J(omega_sum) + J(omega_diff) + 3*J(omegaN)) + (2/15)*(c**2)*J(omegaN)

def R2(x, amps, taus):
    #def J(omega): return j(omega, amp1, amp2, t1, t2, t3)
    def J(omega): return jj(omega, amps, taus)
    field = x * 1e6 / (gamma1H/(2*np.pi))
    omegaN = gamma15N * field
    omegaH = gamma1H * field
    omega_sum = omegaH + omegaN
    omega_diff = omegaH - omegaN
    c = tCSA * omegaN
    return ((d**2)/20)*(6.0*J(omega_sum) + J(omega_diff) + 3*J(omegaN) + 6*J(omegaH) + 4*J(0)) + ((c**2)/45)*(3*J(omegaN) + 4*J(0))

def NOE(x, amps, taus):
    #def J(omega): return j(omega, amp1, amp2, t1, t2, t3)
    def J(omega): return jj(omega, amps, taus)
    field = x * 1e6 / (gamma1H/(2*np.pi))
    omegaN = gamma15N * field
    omegaH = gamma1H * field
    omega_sum = omegaH + omegaN
    omega_diff = omegaH - omegaN
    return 1.0 + (d**2)/(10*R1(x, amp1, amp2, t1, t2, t3)) * (gamma1H/gamma15N)*(6*J(omega_sum)-J(omega_diff))

def etaXY(x, amps, taus):
    #def J(omega): return j(omega, amp1, amp2, t1, t2, t3)
    def J(omega): return jj(omega, amps, taus)
    field = x * 1e6 / (gamma1H/(2*np.pi))
    omegaN = gamma15N * field
    omegaH = gamma1H * field
    omega_sum = omegaH + omegaN
    omega_diff = omegaH - omegaN
    c = tCSA * omegaN
    return -(1/15)*d*c*P2(np.cos(theta))*(3*J(omegaN) + 4*J(0))

def etaZ(x, amps, taus):
    #def J(omega): return j(omega, amp1, amp2, t1, t2, t3)
    def J(omega): return jj(omega, amps, taus)
    field = x * 1e6 / (gamma1H/(2*np.pi))
    omegaN = gamma15N * field
    omegaH = gamma1H * field
    omega_sum = omegaH + omegaN
    omega_diff = omegaH - omegaN
    c = tCSA * omegaN
    return -(1/15)*d*c*P2(np.cos(theta))*6*J(omegaN)
'''
