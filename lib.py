from __future__ import division

import math
import re

import numpy as np

from scipy.integrate import ode
import warnings
warnings.filterwarnings("ignore")


def fcomp(x, t, alpha, mu, K, delta=0.0):
    T, C = x
    Ceff = C/(T+C+K)
    return [alpha*T*Ceff-delta*T, -mu*C]

def fcompfull(x, t, alpha, mu, K, delta=0.0):
    T, C = x
    B = 0.5*(T+C+K - ((T+C+K)**2 - 4*T*C)**.5)
    return [alpha*B-delta*T, -mu*C]

def fsaturation(x, t, alpha, mu, K, delta=0.0):
    T, C = x
    Ceff = C/(K+C)
    return [alpha*T*Ceff-delta*T, -mu*C]

def compbindingtwo(T1, T2, C, K1, K2):
    """Returns the solution to the cubic equation for the bound fraction of type 1."""
    #TODO: solve special case of K1 = K2 properly
    # need explicit float conversion (does not work for numpy floats)
    T1 = float(T1)
    T2 = float(T2)
    C = float(C)
    K1 = float(K1)
    K2 = float(K2)
    a = K1-K2
    b = K1*T2+K2*T1-(K1-K2)*(K1+C+T1)
    c = -T1*(K1*T2+K2*T1+K1*K2+C*(2*K2-K1))
    d = K2*C*T1**2
    P = b**2 - 3*a*c
    Q = 2*b**3 -9*a*b*c+27*a**2 *d
    if K2 > K1:
        Sigma = ((Q+1j*(4*P**3-Q**2)**0.5)/2.0)**(1/3.)
    else:
        Sigma = ((Q-1j*(4*P**3-Q**2)**0.5)/2.0)**(1/3.)
    B = -(b+((-1+3**0.5*1j)*Sigma).real)/(3.0*a)
    if np.isnan(B):
        raise Exception('B is nan')
    return B

def ftwospecifities(x, t, alpha, mu, K1, K2, delta):
    "rhs of dynamics with explicit solution for competition of two types of T cells"
    T1, T2, C = x

    B1 = compbindingtwo(T1, T2, C, K1, K2)
    B2 = compbindingtwo(T2, T1, C, K2, K1)

    dT1dt=alpha*B1-delta*T1
    dT2dt=alpha*B2-delta*T2
    dCdt=-mu*C
    return [dT1dt, dT2dt, dCdt]

def odeint(f, y0, t, args, Dfun=None, integrator='dopri5', **kwargs):
    """Provides a odeint-like wrapper around the other ode routines from scipy."""
    def f_ode(t, y):
        return f(y, t, *args)
    odeint = ode(f_ode, jac=Dfun)
    odeint.set_integrator(integrator, **kwargs)
    odeint.set_initial_value(y0, t[0])
    ys = np.empty((len(t), len(y0)))
    for i, ti in enumerate(t):
        y = odeint.integrate(ti)
        ys[i] = y
    return ys


def _split(number):
    """ Split a number in python scientific notation in its parts.
        
        @return value and exponent of number

    """
    return re.search(r'(-?[0-9].[0-9]*)(?:e\+?)(-?[0-9]*)', number).groups()

def str_quant(u, uerr, scientific=False):
    """ Make string representation in nice readable format
    
        >>> str_quant(0.0235, 0.0042, scientific = True)
        '2.4(5) \\\cdot 10^{-2}'
        >>> str_quant(1.3, 0.4)
        '1.3(4)'
        >>> str_quant(8.4, 2.3)
        '8(3)'
        >>> str_quant(-2, 0.03)
        '-2.00(3)'
	>>> str_quant(1432, 95, scientific = True)
	'1.43(10) \\\cdot 10^{3}'
	>>> str_quant(1402, 95, scientific = True)
	'1.40(10) \\\cdot 10^{3}'
        >>> str_quant(6.54, 0.14)
        '6.54(14)'
        >>> str_quant(0.8, 0.2, scientific=False)
        '0.8(2)'
        >>> str_quant(45.00, 0.05, scientific=False)
        '45.00(5)'

    """
    # preformatting
    number = format(float(u), "e")
    error = format(float(uerr), "e")
    numberValue, numberExponent = _split(number) 
    errorValue, errorExponent = _split(error)
    numberExponent, errorExponent = int(numberExponent), int(errorExponent)    

    # Precision = number of significant digits
    precision = numberExponent - errorExponent
    # make error
    if errorValue.startswith("1"):
        precision += 1
        errorValue = float(errorValue) * 10  # roundup second digit
    error = int(math.ceil(float(errorValue))) # roundup first digit

    # number digits after point (if not scientific)
    nDigitsAfterPoint = precision - numberExponent
    # make number string
    if scientific:
        number = round(float(numberValue), precision)
        if precision == 0:
            number = int(number)
    else:
        number = round(float(numberValue) * 10**numberExponent, nDigitsAfterPoint)
        if nDigitsAfterPoint == 0:
            number = int(number)
    numberString = str(number)

    # pad with 0s on right if not long enough
    if "." in numberString and not scientific:
        length = numberString.index(".") + nDigitsAfterPoint + 1
        numberString = numberString.ljust(length, "0")
    if scientific:
        length = numberString.index(".") + precision + 1
        numberString = numberString.ljust(length, "0")
    
    if scientific and numberExponent != 0:
        outputString = "%s(%d) \cdot 10^{%d}" % (numberString, error, numberExponent)
    else:
        outputString = "%s(%d)" % (numberString, error)

    return outputString

def str_quant_array(array, **kwargs):
    """ Input array, output mean(se) as string"""
    array = np.asarray(array)
    mean = np.mean(array)
    se = np.std(array, ddof=1)/len(array)**.5
    return str_quant(mean, se, **kwargs)


