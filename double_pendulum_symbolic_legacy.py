import numpy as np
import sympy as sy
from sympy import *
from sympy.matrices import *
from sympy.utilities.lambdify import lambdify, implemented_function
import itertools
from functools import partial
import math
import pickle

picklefile_ode = 'ode.pkl'
picklefile_lin = 'lin.pkl'

def get_pickled(fname):
    try:
        with open('fname') as f: 
            res = pickle.load(f)
    except (IOError, EOFError):
        res = dict()

    return res

def set_pickled(ob, fname):
    with open('fname', 'wb') as f: 
        pickle.dump(ob, f)

def get_ode_fcn_floating_dp(g_, a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_):
    """ Returns a function object that can be called with fnc(t,x, u),
    where x = [q1, q2, q3, q4, qd1, qd2, qd3, qd4], and tau = [tau1, tau2, tau3, tau4], is the torques at each 
    joint respectively. The function implements the ode for the floating inverted
    double pendulum.

    For faster return of already constructed models, the set of parameters
    are checked against a pickled dict.
    """
    

    params = (g_, a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_)

    
    tau1, tau2, tau3, tau4 = sy.symbols('tau1, tau2, tau3, tau4')
    q1, q2, q3, q4, qd1, qd2, qd3, qd4 = symbols('q1, q2, q3, q4, qd1, qd2, qd3, qd4', real=True)

    s1 = sy.sin(q1); s1_ = Symbol('s1')
    c1 = sy.cos(q1); c1_ = Symbol('c1')
    s2 = sy.sin(q2); s2_ = Symbol('s2')
    c2 = sy.cos(q2); c2_ = Symbol('c2')
    s12 = sy.sin(q1+q2); s12_ = Symbol('s12') 
    c12 = sy.cos(q1+q2); c12_ = Symbol('c12') 



    odes = get_symbolic_ode_floating_dp(params)

    # Substitute functions for faster evaluation
    odes = odes.subs(s1, s1_).subs(c1,c1_).subs(s2,s2_).subs(c2,c2_).subs(s12,s12_).subs(c12,c12_)

    lmb = lambdify( (q1, q2, q3, q4, q1dot, q2dot, q3dot, q4dot, s1_, c1_, s2_, c2_, s12_, c12_, tau1, tau2, tau3, tau4), odes)
    return partial(lambda_ode, lambdafunc=lmb)


def get_symbolic_ode_floating_dp(params):
    """
    Will generate ode on symbolic form:
        qdd = H^{-1} (-C(q,qd)*qd - G(q) - S*tau)

    """
     existing_odes = get_pickled(picklefile_ode)

     if params in existing_odes:
         odes = existing_odes[params]
     else:
         (g_, a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_) = params

         tau1, tau2, tau3, tau4 = sy.symbols('tau1, tau2, tau3, tau4')
         q1, q2, q3, q4, qd1, qd2, qd3, qd4 = symbols('q1, q2, q3, q4, qd1, qd2, qd3, qd4', real=True)
         g, a1, L1, m1, I1, a2, L2, m2, I2 = symbols('g, a1, L1, m1, I1, a2, L2, m2, I2')
         
         H, C, G, S = pendulum_ode_manipulator_form_floating_dp()
             
         # Substitute the given parameters
         H = H.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, g_)
         C = C.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, g_)
         G = G.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, g_)

         # Invert symbolically
         tau = Matrix([tau1, tau2, tau3, tau4])
         qdot = Matrix([qd1, qd2, qd3, qd4])
        
         odes = H.LUsolve(S*tau-C*qdot-G)
        
         # pickle
         existing_odes[params] = odes
         set_pickled(existing_odes, picklefile_ode)
         
     return odes
    
def get_ode_fcn(a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_, useRT=True):
    """ Returns a function object that can be called with fnc(t,x, u),
    where x = [th1,th2], and u = [tau1, tau2], is the torques at each 
    joint respectively. The function implements the ode for the inverted
    double pendulum.

    For faster return of already constructed models, the set of parameters
    are checked against a pickled dict.
    """
    

    params = (a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_, useRT)

    
    u1 = Symbol('u1')
    u2 = Symbol('u2')
    th1 = Symbol('th1')
    th2 = Symbol('th2')
    th1dot = Symbol('th1dot')
    th2dot = Symbol('th2dot')

    s1 = sin(th1); s1_ = Symbol('s1')
    c1 = cos(th1); c1_ = Symbol('c1')
    s2 = sin(th2); s2_ = Symbol('s2')
    c2 = cos(th2); c2_ = Symbol('c2')
    s12 = sin(th1+th2); s12_ = Symbol('s12') 
    c12 = cos(th1+th2); c12_ = Symbol('c12') 



    odes = get_symbolic_ode(params)

    # Substitute functions for faster evaluation
    odes = odes.subs(s1, s1_).subs(c1,c1_).subs(s2,s2_).subs(c2,c2_).subs(s12,s12_).subs(c12,c12_)

    lmb = lambdify((th1,th2,th1dot, th2dot, s1_, c1_, s2_, c2_, s12_, c12_, u1, u2), odes)
    return partial(lambda_ode, lambdafunc=lmb)


def get_symbolic_ode(params):
         
     existing_odes = get_pickled(picklefile_ode)

     if params in existing_odes:
         odes = existing_odes[params]
     else:
         (a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_, useRT) = params

         u1 = Symbol('u1')
         u2 = Symbol('u2')
         th1dot = Symbol('th1dot')
         th2dot = Symbol('th2dot')
         a1, L1, m1, I1, a2, L2, m2, I2, g = symbols('a1, L1, m1, I1, a2, L2, m2, I2, g')
         
         if useRT: # Expressions taken from Tedrake 2009, explicit form
             H,C,G = pendulum_ode_manipulator_form_RT()
         else:
             H,C,G = pendulum_ode_manipulator_form()
             
         # Substitute the given parameters
         H = H.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, 9.825)
         C = C.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, 9.825)
         G = G.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, 9.825)

         #return partial(manipulator_ode,H=H, C=C, G=G)

         # Invert symbolically
         u = Matrix([u1, u2])
         qdot = Matrix([th1dot, th2dot])
        
         odes = H.LUsolve(u-C*qdot-G)
        
         # pickle
         existing_odes[params] = odes
         set_pickled(existing_odes, picklefile_ode)
         
     return odes
    
def lambda_ode(x, u, lambdafunc):
    th1 = x[0]
    th2 = x[1]
    th1dot = x[2]
    th2dot = x[3]
    s1 = math.sin(th1) 
    c1 = math.cos(th1)
    s2 = math.sin(th2)
    c2 = math.cos(th2)
    s12 = math.sin(th1+th2)
    c12 = math.cos(th1+th2)
    
    return np.ravel(lambdafunc(th1, th2, th1dot, th2dot, s1, c1, s2, c2, s12, c12, u[0], u[1]))
    
def manipulator_ode(x, u, H, C, G):
    """ Manipulator ode. Will evaluate assuming x = [th1,th2, th1dot, th2dot] """
    th1 = Symbol('th1'); th1_ = x[0]
    th2 = Symbol('th2'); th2_ = x[1]
    th1dot = Symbol('th1dot'); th1dot_ = x[2]
    th2dot = Symbol('th2dot'); th2dot_ = x[3]
    s1 = Symbol('s1'); s1_ = math.sin(th1_) 
    c1 = Symbol('c1'); c1_ = math.cos(th1_)
    s2 = Symbol('s2'); s2_ = math.sin(th2_)
    c2 = Symbol('c2'); c2_ = math.cos(th2_)
    s12 = Symbol('s12'); s12_ = math.sin(th1_+th2_)
    c12 = Symbol('c12'); c12_ = math.cos(th1_+th2_)
    
    H = H.subs(th1,th1_).subs(th2,th2_).subs(th1dot,th1dot_).subs(th2dot,th2dot_)
    H = H.subs(s1,s1_).subs(c1,c1_).subs(s2,s2_).subs(c2,c2_).subs(s12,s12_).subs(c12,c12_)
    C = C.subs(th1,th1_).subs(th2,th2_).subs(th1dot,th1dot_).subs(th2dot,th2dot_)
    C = C.subs(s1,s1_).subs(c1,c1_).subs(s2,s2_).subs(c2,c2_).subs(s12,s12_).subs(c12,c12_)
    G = G.subs(th1,th1_).subs(th2,th2_).subs(th1dot,th1dot_).subs(th2dot,th2dot_)
    G = G.subs(s1,s1_).subs(c1,c1_).subs(s2,s2_).subs(c2,c2_).subs(s12,s12_).subs(c12,c12_)

    #2/0

    # Convert to numpy and invert
    Hnp = to_np(H)
    b = -np.dot(to_np(C), np.array([th1dot_, th2dot_])).flatten() - to_np(G).flatten() + u
    
    return np.linalg.solve(Hnp, b)



def to_np(A):
    """ Converts sympy matrix A to numpy matrix """
    shapeA = A.shape
    Anp = np.zeros(shapeA)
    for i in range(0,shapeA[0]):
        for j in range(0,shapeA[1]):
            Anp[i,j]=sympy.N(A[i,j])
    return Anp



def get_linearized_ode(a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_):
    """ Returns the system matrix A for a linear state space model with the 
    provided values substituted.
    """

    params = (a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_)

    existing_odes = get_pickled(picklefile_lin)
    if params in existing_odes:
        A,B = existing_odes[params]
    else:
        Asymb, Bsymb = linearize_ode(th1_=np.pi)
        a1, L1, m1, I1, a2, L2, m2, I2, g = symbols('a1, L1, m1, I1, a2, L2, m2, I2, g')
        A = Asymb.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, 9.825)
        B = Bsymb.subs(a1, a1_).subs(L1, L1_).subs(m1, m1_).subs(I1, I1_).subs(a2, a2_).subs(L2, L2_).subs(m2, m2_).subs(I2, I2_).subs(g, 9.825)

        A = to_np(A)
        B = to_np(B)
    
        existing_odes[params] = (A,B)
        set_pickled(existing_odes, picklefile_lin)

    return A,B

def is_controllable(B):
    """ Checks if the linearized system about the vertical position is controllable with
    the provided B matrix
    """

    A, BB = get_linearized_ode(1,2,1,1,1,2,1,1)

    B = np.dot(BB, B)
    C = B.copy()
    AA = A.copy()
    for i in range(1,4):
        C = np.hstack((C, np.dot(AA,B)))
        AA = np.dot(AA,A)

    cdet = np.linalg.det(C)
    print cdet
    return (cdet != 0)

def is_controllable_symbolic(B):
    """ Checks if the linearized system about the vertical position is controllable with
    the provided B matrix
    """

    A, BB = linearize_ode()
    #    2/0
    B = BB*B
    C = B[:,:]
    AA = A[:,:]
    for i in range(1,4):
        C = C.row_join(AA*B)
        AA = AA*A

    2/0
    cdet = C.det()
    print cdet
    return (cdet != 0)

def linearize_ode(th1_=0, th2_=0, th1dot_=0, th2dot_=0):
    """ Returns a state space model of the dynamics linearized about the 
    given angles
    """

    th1, th2, th1dot, th2dot = symbols('th1, th2, th1dot, th2dot')
    H, C, G = pendulum_ode_manipulator_form_RT()

    dG = G.jacobian((th1,th2))
    Hinv = H.inv()
    A00 = zeros(2,2)
    A01 = eye(2)
    A10 = -Hinv*dG
    A11 = -Hinv*C
    A = A00.row_join(A01).col_join(A10.row_join(A11))
    
    #2/0
    B = zeros(2,2).col_join(Hinv)

    return A.subs(th1,th1_).subs(th2,th2_).subs(th1dot,th1dot_).subs(th2dot,th2dot_), \
        B.subs(th1,th1_).subs(th2,th2_).subs(th1dot,th1dot_).subs(th2dot,th2dot_)

def pendulum_ode():
    """ Returns the equations of motion for the double pendulum
    """
    a1, L1, m1, I1, a2, L2, m2, I2 = symbols('a1, L1, m1, I1, a2, L2, m2, I2')
    g = Symbol('g')
    th1, th2, th1dot, th2dot = symbols('th1, th2, th1dot, th2dot')

    s1 = sin(th1)
    s2 = sin(th2)
    c1 = cos(th1)
    c2 = cos(th2)
    c21 = cos(th2-th1)

    I1p0 = I1 + m1*a1**2
    I2p1 = I2 + m2*a2**2

    f1 = a2*L1*c21
    H2th2 = I2 + m2*(a2**2 + f1)
    H2th1 = m2*(L1**2 + f1)
    H2rest = m2 * ( (L1*s1 + a2*s2) * (L1*th1dot**2*c1 + a2*th2dot**2*c2) \
                        + (L1*c1 + a2*c2) * (L1*th1dot**2*s1 - a2*th2dot**2*s2) )

    A = Matrix([[I1p0+H2th1, H2th2], [-I2p1+m2*f1, I2p1]])

    b = Matrix([m1*a1*g*s1 + m2*g*(L1*s1+a2*s2) - H2rest  + tau1, 
                m2*a2*g*s2 + tau2])

    dx1 = A.LUsolve(b)

    dx= Matrix([th1dot, th2dot]).col_join(dx1)
    
    #print dx
    #2/0
    return dx


def small_angles_approximation():
    """ Returns the equations of motion for the double pendulum using 
    the small angle approximations for both joints.
    """
    a1, L1, m1, I1, a2, L2, m2, I2 = symbols('a1, L1, m1, I1, a2, L2, m2, I2')
    g = Symbol('g')
    th1func = Function('th1')
    th2func = Function('th2')
    t = Symbol('t')
    th1 = th1func(t)
    th2 = th2func(t)
    th1dot = diff(th1,t)
    th2dot = diff(th2,t)

    I1p0 = I1 + m1*a1**2
    I2p1 = I2 + m2*a2**2

    f1 = a2*L1
    H2th2 = I2 + m2*(a2**2 + f1)
    H2th1 = m2*(L1**2 + f1)
    H2rest = m2 * ( (L1*th1 + a2*th2) * (L1*th1dot**2 + a2*th2dot**2) \
                        + (L1 + a2) * (-L1*th1dot**2*th1 - a2*th2dot**2*th2) )

    A = Matrix([[I1p0+H2th1, H2th2], [-I2p1+m2*f1, I2p1]])

    b = Matrix([m1*a1*g*th1 + m2*g*(L1*th1+a2*th2) - H2rest, 
                m2*a2*g*th2])

    dx = A.LUsolve(b)

    print dx
    return dx


def test_pendulum_ode_manipulator_form():
    """ Compares computed odes with those of Ross Tedrake 2009 """

    a1, L1, m1, I1, a2, L2, m2, I2 = symbols('a1, L1, m1, I1, a2, L2, m2, I2')
    g = Symbol('g')
    th1, th2, th1dot, th2dot = symbols('th1, th2, th1dot, th2dot')

    I11 = I1 + m1*a1**2 # Moment of inertia wrt pivot point
    I22 = I2 + m2*a2**2 # Moment of inertia wrt joint

    s1 = sin(th1)
    c1 = cos(th1)
    s2 = sin(th2)
    c2 = sin(th2)
    s12 = sin(th1+th2)
    c12 = cos(th1+th2)

    H, C, G = pendulum_ode_manipulator_form(False)
    Ht, Ct, Gt = pendulum_ode_manipulator_form_RT()

    for (expct, got, txt) in ((Ht, H, 'H'), (Ct, C, 'C'), (Gt, G, 'G')):
        for (e, g) in itertools.izip(expct, got):
             eq = trigsimp((e-g).expand())
             if not eq.equals(numbers.Zero()):
                 # Could still be the same. Try to solve
                 sol = solve(eq)
                 if sol[0].values()[0] != numbers.Zero():
                     print "Error in " + txt
                     print "Element " 
                     print g

    2/0

def pendulum_ode_manipulator_form_floating_dp():
    """
    Equations of motion on  manipulator form for floating double pendulum. The manipulator form
    is 
        H * qdd + C(q,qd)*qd + G(q) = S*tau
    where the matrix S picks the actuated degrees of freedom. 

    Definitions of the generalized coordinates:
        :q1: Angle between link one and vertical. Positive rotation is about the y-axis, which points into the plane.
        :q2: Angle between link one and two.
        :q3: Position in x-direction (horizontal) of the base joint (ankle joint).
        :q4: Position in z-direction (vertical) of the base joint (ankle joint).

    Returns:
        Tuple: (H, C, G, S) 

    There are a number of symbols present in the return values. These are:
        :qi: generalized coordinate i
        :qdi: generalized velocity i
        :g: the local magnitude of the gravitational field
        :m1: the mass of link 1
        :m2: the mass of link 2
        :L1: the length of link 1
        :L2: the length of link 2
        :I1: the moment of inertia of link 1 wrt to its CoM
        :I2: the moment of inertia of link 2 wrt to its CoM
        :a1: the position of CoM of link 1 along link 1 from base joint to next joint. 
                 0 <= a1 <= 1. a1=1 means the CoM is at the next joint.
        :a2: the position of CoM along link 1
    
    :Date: 2016-04-18

    """

    q1, q2, q3, q4, qd1, qd2, qd3, qd4 = symbols('q1, q2, q3, q4, qd1, qd2, qd3, qd4', real=True)
    qdd1, qdd2, qdd3, qdd4 = symbols('qdd1, qdd2, qdd3, qdd4', real=True)
    g, m1, m2, l1, l2, I1, I2, a1, a2 = symbols('g, m1, m2, l1, l2, I1, I2, a1, a2', real=True, positive=True)
    tau1, tau2, Fx, Fy = symbols('tau1, tau2, Fx, Fy', real=True)


    q = Matrix([[q1],[q2],[q3], [q4]])
    qd = Matrix([[qd1],[qd2],[qd3], [qd4]])
    qdd = Matrix([[qdd1],[qdd2],[qdd3], [qdd4]])
    tau = Matrix([])
    c1 = sy.cos(q1)
    s1 = sy.sin(q1)
    c12 = sy.cos(q1+q2)
    s12 = sy.sin(q1+q2)

    p1 = sy.Matrix([[q3 +a1*l1*s1],[q4+a1*l1*c1]])
    p2 = sy.Matrix([[q3 +l1*s1 + a2*l2*s12],[q4+l1*c1 + a2*l2*c12]])
    V1 = sy.Matrix([[a2*l1*c1, 0, 1, 0],[-a2*l1*s1, 0, 0, 1]])
    pd1 = V1*qd
    V2 = sy.Matrix([[l1*c1+a2*l2*c12, a2*l2*c12, 1, 0 ],[-l1*s1-a2*l2*s12, -a2*l2*s12, 0, 1]])
    pd2 = V2*qd
    Omega1 = sy.Matrix([[1, 0, 0 , 0]])
    w1 = Omega1*qd
    Omega2 = sy.Matrix([[1, 1, 0 , 0]])
    w2 = Omega2*qd

    H = m1*V1.T*V1 + m2*V2.T*V2 + I1*Omega1.T*Omega1 + I2*Omega2.T*Omega2

    T = 0.5*qd.T*H*qd
    U = sy.Matrix([m1*g*p1[1] + m2*g*p2[1]])

    # The equations of motion on manipulator form
    C = sy.zeros(4,4)
    G = sy.zeros(4,1)

    for i in range(4):
        qi = q[i]
        Hi = H[:,i]
        Gammai = Hi.jacobian(q)*qd
        #ddtdLdqidot = Hi.T*qdd + Gammai.T*qd
        dHdqi = H.diff(qi)
        Di = 0.5*dHdqi*qd
        Gi = U.diff(qi)
        #dLdqi = Di.T*qd - Gi
        #lhs1 = ddtdLdq1dot - dLdq1
        # Form the terms for the manipulator form
        Ci = Gammai - Di
        C[i,:] = Ci.T
        G[i] = Gi

    S = sy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    return (H,C,G,S)

def pendulum_ode_manipulator_form_RT():

    a1, L1, m1, I1, a2, L2, m2, I2 = symbols('a1, L1, m1, I1, a2, L2, m2, I2')
    g = Symbol('g')
    th1 = Symbol('th1')
    th2 = Symbol('th2')
    th1dot = Symbol('th1dot')
    th2dot = Symbol('th2dot')

    I11 = I1 + m1*a1**2 # Moment of inertia wrt pivot point
    I22 = I2 + m2*a2**2 # Moment of inertia wrt joint

    s1 = sin(th1)
    c1 = cos(th1)
    s2 = sin(th2)
    c2 = cos(th2)
    s12 = sin(th1+th2)
    c12 = cos(th1+th2)

    Ht = Matrix([[I11 + I22 + m2*L1**2 + 2*m2*L1*a2*c2, 
                  I22 + m2*L1*a2*c2],
                 [I22 + m2*L1*a2*c2, I22]])

    Ct = Matrix([[-2*m2*L1*a2*s2*th2dot, -m2*L1*a2*s2*th2dot],
                 [m2*L1*a2*s2*th1dot, 0]])

    Gt = Matrix([(m1*a1 + m2*L1)*g*s1 + m2*g*a2*s12, m2*g*a2*s12])

    return Ht, Ct, Gt


def pendulum_ode_manipulator_form(downIsPI=True):
    """ Computes the pendulum ode on manipulator form
          H(q) qddot + C(q,qdot) + G(q) = B(q) u
    using the Lagrangian formulation.
    Change to other definition of angles: th1 is the angle of the first link to the
    vertical, and is zero when the link is upright. th2 is the angle at the joint. 
    It is zero when the joint is straight.
    """


    a1, L1, m1, I1, a2, L2, m2, I2 = symbols('a1, L1, m1, I1, a2, L2, m2, I2')
    g = Symbol('g')
    th1func = Function('th1')
    th2func = Function('th2')
    t = Symbol('t')
    th1 = th1func(t)
    th2 = th2func(t)
    th1dot = diff(th1,t)
    th2dot = diff(th2,t)
    th1ddot = diff(th1dot,t)
    th2ddot = diff(th2dot,t)
    th1_ = Symbol('th1')
    th2_ = Symbol('th2')
    th1dot_ = Symbol('th1dot')
    th2dot_ = Symbol('th2dot')
    th1ddot_ = Symbol('th1ddot')
    th2ddot_ = Symbol('th2ddot')


    I11 = I1 + m1*a1**2 # Moment of inertia wrt pivot point
    I22 = I2 + m2*a2**2 # Moment of inertia wrt joint

    s1 = sin(th1)
    c1 = cos(th1)
    s2 = sin(th2)
    c2 = cos(th2)
    s12 = sin(th1+th2)
    c12 = cos(th1+th2)

    x1 = Matrix([-a1*s1, a1*c1]) # Center of mass of first segment
    p1 = Matrix([-L1*s1, L1*c1]) # Position of the joint
    x2 = p1 + Matrix([-a2*s12, a2*c12])
    
    x1dot = Matrix([[-a1*c1*th1dot],[-a1*s1*th1dot]])
    x2dot = Matrix([[-L1*c1*th1dot],[-L1*s1*th1dot]]) \
        + Matrix([[-a2*c12*(th1dot + th2dot)],[-a2*s12*(th1dot + th2dot)]])
    #x1dot = Matrix([diff(x1[0], t), diff(x1[1], t)])
    #x2dot = Matrix([diff(x2[0], t), diff(x2[1], t)])


    # Kinetic energy
    T = 0.5*m1*x1dot.dot(x1dot) + 0.5*m2*x2dot.dot(x2dot) \
        + 0.5*I1*th1dot**2 + 0.5*I2*(th1dot + th2dot)**2
    #T = 0.5*I11*th1dot**2 + 0.5*I2*(th1dot + th2dot)**2 + 0.5*m2*x2dot.dot(x2dot)
    TT = T.subs(th1dot, th1dot_).subs(th2dot,th2dot_).subs(th1,th1_).subs(th2, th2_)


    # Factorize T as T = 0.5 qdot * H * qdot
    Td1 = diff(TT, th1dot_).expand()
    Td2 = diff(TT, th2dot_).expand()
    dd1 = collect(Td1, (th1dot_, th2dot_), evaluate=False) 
    dd2 = collect(Td2,(th1dot_, th2dot_), evaluate=False) 
    qdot = Matrix([th1dot_, th2dot_])
    
    zer = numbers.Zero()
    for k in (th1dot_, th2dot_):
        if k not in dd1.keys():
            dd1[k] = zer
        if k not in dd2.keys():
            dd2[k] = zer


    H = Matrix([[trigsimp(dd1[th1dot_].expand()), trigsimp(dd1[th2dot_].expand())], 
                 [trigsimp(dd2[th1dot_].expand()), trigsimp(dd2[th2dot_].expand())]])

    # Check if ok
    TTT = 0.5*qdot.dot(H.dot(qdot))
    null = trigsimp((TTT-TT).expand()).simplify()
    if not null.equals(numbers.Zero()):
        print "### Error in factorized kinetic energy!"
        2/0

    # Potential energy
    if downIsPI:    
        V = m1*g*x1[1] + m2*g*x2[1]
    else:
        V = -m1*g*x1[1] - m2*g*x2[1]

    # Lagrangian
    L = T-V


    dLdth1dot = diff(L, th1dot)
    ddtdLdth1dot = diff(dLdth1dot,t)
    dLdth1 = diff(L, th1)

    dLdth2dot = diff(L, th2dot)
    ddtdLdth2dot = diff(dLdth2dot,t)
    dLdth2 = diff(L, th2)


    # The euler-lagrange equations
    EL1 = trigsimp((ddtdLdth1dot -dLdth1).expand())
    EL2 = trigsimp((ddtdLdth2dot -dLdth2).expand())

    # Substiute symbols for th1, th1dot, etc 
    EL1 = EL1.subs(th1ddot, th1ddot_).subs(th2ddot, th2ddot_).subs(th1dot, th1dot_).subs(th2dot, th2dot_).subs(th1, th1_).subs(th2,th2_)
    EL2 = EL2.subs(th1ddot, th1ddot_).subs(th2ddot, th2ddot_).subs(th1dot, th1dot_).subs(th2dot, th2dot_).subs(th1, th1_).subs(th2,th2_)

    one = numbers.One()
    
    # Factorize as H*qddot + C*qdot + G


    #H11 = trigsimp(diff(EL1, th1ddot))
    #H12 = trigsimp(diff(EL1, th2ddot))
    C11 = trigsimp(diff(EL1, th1dot_).expand())
    C12 = trigsimp(diff(EL1, th2dot_).expand())
    G1 = trigsimp((EL1 - H[0,0]*th1ddot_ - H[0,1]*th2ddot_ - C11*th1dot_ - C12*th2dot_).expand()).simplify()

    #H21 = trigsimp(diff(EL2, th1ddot))
    #H22 = trigsimp(diff(EL2, th2ddot))
    C21 = trigsimp(diff(EL2, th1dot_))
    C22 = trigsimp(diff(EL2, th2dot_))
    G2 = trigsimp((EL2 - H[1,0]*th1ddot_ - H[1,1]*th2ddot_ - C21*th1dot_ - C22*th2dot_).expand()).simplify()

    #if not H11.equals(H[0,0]):
    #    print "### Error in calculated inertia matrix"
    #if not H12.equals(H[0,1]):
    #    print "### Error in calculated inertia matrix"
    #if not H21.equals(H[1,0]):
    #    print "### Error in calculated inertia matrix"
    #if not H22.equals(H[1,1]):
    #    print "### Error in calculated inertia matrix"

    
    #H = Matrix([[H11,H12], [H21,H22]])
    C = Matrix([[C11, C12], [C21, C22]])
    G = Matrix([G1,G2])

    #Test that calculations are correct
    ELtest = G1 + H[0,0]*th1ddot_ + H[0,1]*th2ddot_ + C[0,0]*th1dot_ + C[0,1]*th2dot_
    null = trigsimp((ELtest-EL1).expand()).simplify()
    if not null.equals(numbers.Zero()):
        print "#### Error in equations of motion"
        2/0
        
    return H, C, G

    

def pendulum_model_diff(L1_v, m1_v, I1_v, L2_v, m2_v, I2_v):
    a1, L1, m1, I1, a2, L2, m2, I2 = symbols('a1, L1, m1, I1, a2, L2, m2, I2')
    g = Symbol('g')
    th1 = Function('th1')
    th2 = Function('th2')

    x1 = Matrix([[-a1*sin(th1(t))],[a1*cos(th1(t))]]) # Center of mass of first segment
    p1 = Matrix([[-L1*sin(th1(t))],[L1*cos(th1(t))]]) # Position of the joint
    x2 = p1 + Matrix([[-L2*sin(th1(t))],[L2*cos(th1(t))]])
    
    x1dot = Matrix([[diff(x1[0,0], t)], [diff(x1[1,0],t)]])
    x2dot = Matrix([[diff(x2[0,0], t)], [diff(x2[1,0],t)]])

    # Kinetic energy
    T = 0.5*m1*x1dot.dot(x1dot) + 0.5*m2*x2dot.dot(x2dot) \
        + 0.5*I1*diff(th1(t),t)**2 + 0.5*I2*diff(th2(t),t)**2

    # Potential energy
    V = m1*g*a1*cos(th1(t)) + m2*g*(L1*cos(th1(t)) + a2*cos(th1(t)))

    # Lagrangian
    L = T-V
    

def get_cc_ode_fcn(a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_, useRT=True):
    """ Returns a list of four strings that defines the c-functions of the ode for the 
    double pendulum.

    For faster return of already constructed models, the set of parameters
    are checked against a pickled dict.
    """

    params = (a1_, L1_, m1_, I1_, a2_, L2_, m2_, I2_, useRT)

    
    u1 = Symbol('u1')
    u1_ = Symbol('u1( th1(t-tau), th2(t-tau), th1dot(t-tau), th2dot(t-tau) )')
    u2 = Symbol('u2')
    u2_ = Symbol('u2( th1(t-tau), th2(t-tau), th1dot(t-tau), th2dot(t-tau) )')
    th1 = Symbol('th1')
    th2 = Symbol('th2')
    th1dot = Symbol('th1dot')
    th2dot = Symbol('th2dot')


    odes = get_symbolic_ode(params)
    odes = odes.subs(u1,u1_).subs(u2,u2_)

    ccodes = ['th1dot', 'th2dot']
    ccodes.append(ccode(odes[0]))
    ccodes.append(ccode(odes[1]))

    return ccodes

    


