"""
Code for simulating a model of an actuated double pendulum 

Revisions
2016-04-07
    Complete rewrite based on matplotlib. See 
       http://matplotlib.org/1.4.1/examples/animation/double_pendulum_animated.html
    
"""
import matplotlib
#matplotlib.use("QTAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import os, sys
import random
from functools import partial

from traits.api import HasTraits, Float, Button, Enum, String, Array
from traitsui.menu import Action
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from traitsui.api import *
import math
import numpy as np
import  scipy.integrate 

from timeseriesplot import TimeSeriesPlot
import double_pendulum_symbolic as dps

class DoublePendulumAnimation:
    """
    Class to animate a planar double pendulum. The class is instantiated with a desired size::

       animation = DoublePendulum(width=800, height=400, fps=60)

    The animation must first be intialized. This is done using init_dp:
    
       animation.init_dp(lenLink1, lenLink2, angle1, angl2)

    To run an animation, a signal source object needs to be passed which can generate 
    the sequence of joint angles and input signals. For a completely static animation::
       
       class StaticSource:
           def get_next_input(self, t, dt):
               return 0.0
           def get_next_state(self, t, dt):
               return np.array([0.0, 0.0])
           def get_set_point(self, t):
               return np.array([0.0, 0.0])

       animation.run_sim(StaticAnim(), tmax=10)
    """
    def __init__(self, pendulumModel, fps=50.0):

        self.pendulumModel = pendulumModel
        self.fps = fps
        self.height= 800
        self.width = 2*self.height
        self.dpi = 100
        self.fig = plt.figure(figsize=(self.width/self.dpi, self.height/self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_axes([0.03, 0.05, 0.45, 0.9], axisbg='black', autoscale_on=False)
        self.ax.grid(True, color='gray')
        plt.axis('equal')
        self.pendulum, = self.ax.plot([], [], 'o-', lw=6, color=(0.7, 0.1, 0.1))
        self.comPlot, = self.ax.plot([], [], 'o', lw=6, color=(0.1, 0.7, 0.1))
        self.FxVector = self.ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0.8, 0), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color=(0.8, 0.8, 0.8)))
        self.FzVector = self.ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0, -0.8), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color=(0.8, 0.8, 0.8)))
        self.forceScale = 1.0/500.0 # Force vector of 500N shown as a unit length (1m) vector
        self.grfVector = self.ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0, -0.8), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color=(0.8, 0.8, 0.8), linestyle='dashed'))

        rect = [0.5, 0.05, 0.47, 0.9] # The rect available for the rest of the plots
        self.timeseriesplot = TimeSeriesPlot(self.fig, rect )
                                        
        self.animated = self.timeseriesplot.animated
        self.animated.append(self.pendulum)
        self.animated.append(self.comPlot)
        self.animated.append(self.FxVector)
        self.animated.append(self.FzVector)
        self.animated.append(self.grfVector)

        self.init_plot()
        
    def init_dp(self, q1, q2, q3, q4, Fx=0, Fz=0, My=0):
        """ Called by DP model when all parameters are set by the user, and the simulation 
        is about to start.
        """
        length1 = self.pendulumModel.L1
        length2 = self.pendulumModel.L2

        dplength = length1+length2

        self.root_pos = [q3, q4]
        
        self.ax.set_xlim(1.2*np.array([-dplength, dplength]))
        self.ax.set_ylim(0.1*dplength + 1.2*np.array([-dplength, dplength]))

        self.draw_pendulum(np.array([q1, q2, q3, q4]))

        if Fz > 0:
            dx = -float(My)/float(Fz)
        else:
            dx = 0

        self.FxVector.xy = (q3+dx, q4)
        self.FxVector.set_position((q3+dx-Fx*self.forceScale, q4))
        self.FzVector.xy = (q3+dx, q4)
        self.FzVector.set_position((q3+dx, q4 - Fz*self.forceScale))
        self.grfVector.set_position((q3+dx, q4))
        self.grfVector.xy = (q3+dx+Fx*self.forceScale, q4 + Fz*self.forceScale)

        plt.draw()

    def reset(self):
        self.fig.clear()
        self.init_plot()
        self.master.update()

    def init_plot(self):
        ts = []
        ts.append(dict(name='Inputs',
                       ylim=(-10, 10),
                       timeseries=[ dict(label='u1', linewidth=2, color=(1,1,0)), 
                                    dict(label='u2', linewidth=2, color=(0.6,0.4,0))]))
        ts.append(dict(name='Generalized coordinates',
                       ylim=(-4,4),
                       timeseries=[ dict(label='q1', linewidth=2, color=(0,1,1)), 
                                    dict(label='q2', linewidth=2, color=(0,0.4,0.6)),
                                    dict(label='q3', linewidth=1, color=(0.2,0.6,0.4)),
                                    dict(label='q4', linewidth=1, color=(0.2,1,0.6))
                       ]))
        ts.append(dict(name='Ground reaction forces',
                       ylim=(-800,800),
                       timeseries=[ dict(label='Fx', linewidth=2, color=(1,0,1)), 
                                    dict(label='Fz', linewidth=2, color=(0.6, 0, 0.4)),
                                    dict(label='My', linewidth=1, color=(0.4,0.,0.6)),
                       ]))
        #ts.append(dict(name='Power input', 
        #               timeseries=[ dict(label='power', linewidth=1, color=(1,0,1))]))
        #ts.append(dict(name='Control cost',
        #               ylim=(0,10),
        #               timeseries=[ dict(label='cost', linewidth=2, color=(1,0,1))]))
        
        self.timeseriesplot.init_plot(ts)
        
    def power(th1dot, th2dot, u1, u2):
        """ Calculates the instanenous power added to (or subtracted from) the system. """
        return (th1dot*u1 + th2dot*u2)

    def init_control_cost(self, costfunction):
        self.cost = 0
        self.costfunction = costfunction 
        self.t = 0

    def update_control_cost(self, x, u, dt):
        """ Computes the recursive least squares of the control cost """
        t = self.t + dt
        self.cost += dt/t*(self.costfunction(x, u) - self.cost)
        self.t = t
        return self.cost

    def draw_pendulum(self, q):
        self.pendulumModel.set_state(q)
        (rootp, jp, endp) = self.pendulumModel.get_joints_and_endpoint()
        (com, com1, com2) = self.pendulumModel.get_CoM()
        
        px = [rootp[0], jp[0], endp[0]]
        py = [rootp[1], jp[1], endp[1]]
        self.pendulum.set_data(px, py)

        cx = [com[0], com1[0], com2[0]]
        cy = [com[1], com1[1], com2[1]]
        self.comPlot.set_data(cx, cy)

        
    def run_sim(self, signalSource, tmax, costfunction = lambda u,x: 0.0, plotTimeSeries = False, loop=False):
        self.init_control_cost(costfunction)
        nFrames = tmax * self.fps
        dt = 1.0 / self.fps

        ani = animation.FuncAnimation(self.fig, self.animate, frames=np.arange(nFrames)*dt,
                                      fargs=(dt, signalSource, costfunction, plotTimeSeries),
                                      interval=dt*1000, blit=True, repeat=loop) #, init_func=self.init)

        plt.show()
        
    def animate(self, t, dt, signalSource, costfunction, plotTimeSeries):

        now = t
    
        u = signalSource.get_next_input(now, dt)
        q = signalSource.get_next_state(now, dt)
        q0 = signalSource.get_set_point(now)
        grf = signalSource.get_next_grf(now, dt)
        

        if np.isscalar(u):
            u1 = u
            u2 = 0
        else:
            u1 = u[0]
            if len(u) > 1:
                u2 = u[1]
            else:
                u2 = 0

        # Update the pendulum        
        self.draw_pendulum(q)

        # Update the force vectors
        # The center of pressure
        Fx = grf[0]
        Fz = grf[1]
        My = grf[2]
        dx = -Myn/Fz
        self.FxVector.xy = (q[2]+dx, q[3])
        self.FxVector.set_position((q[2]+dx-Fx*self.forceScale, q[3]))
        self.FzVector.xy = (q[2]+dx, q[3])
        self.FzVector.set_position((q[2]+dx, q[3] - Fz*self.forceScale))
        self.grfVector.set_position((q[2]+dx, q[3]))
        self.grfVector.xy = (q[2]+dx+Fx*self.forceScale, q[3] + Fz*self.forceScale)
        
        
        if plotTimeSeries:
            dta = {}
            dta['q1'] = q[0]
            dta['q2'] = q[1]
            dta['q3'] = q[2]
            dta['q4'] = q[3]
            dta['u1'] = u1
            dta['u2'] = u2
            dta['Fx'] = Fx
            dta['Fz'] = Fz
            dta['My'] = My

            #dta['cost'] = self.update_control_cost(q-q0, u, dt)
            self.timeseriesplot.append_data(now, dta)
        return self.animated
    
def restart_sim():
    pass
def shutdown_app():
    pass

class DoublePendulum(HasTraits):
    """ Class representing the dynamics of a double pendulum """

    m1 = Float(1.0, label="Mass of link 1")
    L1 = Float(1.0, label="Length of link 1")
    a1 = Float(.5, label="position of CoM of link 1")
    I1 = Float(1/12.0, label="Moment of inertia of link 1")
    m2 = Float(2.0, label="Mass of link 2")
    L2 = Float(1.2, label="Length of link 2")
    a2 = Float(.7, label="position of CoM of link 2")
    I2 = Float(2/12.0, label="Moment of inertia of link 2")
    
    reflexThreshold = Float(-0.1, label="Reflex threshold (in m)")
    reflexAmplitude = Float(100.0, label="Amplitud of actuator torque during reflex")
    reflexDuration = Float(0.05, label="Duration of reflex impulse")
    reflexRefractoryPhase = Float(0.1, label="Refractor phase after reflex impulse")
    
    stiffnessControl = Float(0.0, label="Stiffness factor for impedance control")
    
    delay = Float(.0, label="Time delay in feedback loop")
    actuatorSaturation = Float(-1.0, label="Actuator saturation level (negative => no sat)")

    noiseVar = Float(4.0, label="Noise variance")

    actuate = Enum('Both', 'First only', 'Second only', 
                   label="Which joint to actuate")
    
    feedbackType = Enum('Manually set', 'LQR',
                        label="State feedback")

    stateFeedbackGain = Array(np.float, (2,4), value=np.zeros((2,4)))

    setpoint = Array(np.float, (4,), value=np.array([np.pi, 0, 0, 0]))
    friction = Array(np.float, (2,), value=np.array([0.0, 0.0]))

    # And to choose among the different feedback options

    # Simulation settings.
    x0 = Array(np.float, (4,), value=np.array([np.pi, 0, 0, 0]), label="Initial state")
    tmax = Float(10.0, label="Simulation time")

    # When user press the reset button, then the model is updated
    update_parameters = Button()
    start_simulation = Button()
    reset_animation = Button()

    status = String(label="Status")
    
    # Attributes not set by traitsUI
    rng = random.Random(1234)
    g = 9.8
    sin = math.sin
    cos = math.cos
    # The state is x=[th1, th2, th1dot, th2dot, xe1, xe2],
    # where xe1 and xe2 are the integrated control errors.
    x = [0 for i in range(6)] 
    u = None # Torque input
    v = None # Disturbance
    disturbance = None
    feedback = None
    animation = None
    integrator = None

    # The view used to edit double pendulum parameters
    view = View(HGroup( Group(Item(name="m1"), 
                             Item(name="L1"), 
                             Item(name="a1"), 
                             Item(name="I1"),
                             label="Parameters for link 1",
                         ),
                       Group(Item(name="m2"), 
                             Item(name="L2"), 
                             Item(name="a2"), 
                             Item(name="I2"),
                             label="Parameters for link 2",
                         ),
                       Item(name="friction"),
                       Item(name="noiseVar")
                   ),
                Group(Item(name="reflexThreshold"),
                      Item(name="reflexAmplitude"),
                      Item(name="reflexDuration"),
                      Item(name="reflexRefractoryPhase"),
                      Item(name="stiffnessControl"),
                      Item(name="actuatorSaturation"),
                      Item(name="setpoint"),
                      Item(name="actuate"),
                      Item(name="stateFeedbackGain"),
                      Item(name="feedbackType"),
                      label="Control parameters",
                      show_border=True),
                Group(Item(name="x0"),
                      Item(name="tmax"),
                      label="Simulation parameters",
                      show_border=True),
                HGroup(Item(name="update_parameters", show_label=False), 
                       Item(name="start_simulation", show_label=False),
                       Item(name="reset_animation", show_label=False)),
                Item(name="status"))

    def __init__(self):
        pass

    def set_animation(self, animation):
        self.animation = animation


    def set_trajectory(self, t, qs, grf):
        """
        Sets a trajectory for the pendulum model.
        t should be a (N,) 1D-array of time instants
        qs and grf should be a corresponding sequences (N,4) and ground reaction forces (N,3)
        """
        
        self.timeVector = t
        self.stateTrajectory = qs
        self.grfTrajectory = grf
        
    def set_state(self, q):
        """
        Explicitly sets the state of the pendulum. 
        q must have four elements
        """

        self.currentState = q


    def simulate_trajectory(self, q0, qd0, tmax, dt, controlLaw= lambda x: np.zeros(4)):
        (H_func,f_func) = dps.get_ode_fcn_floating_dp(self.g,
                                            self.a1, self.L1, self.m1, self.I1,
                                            self.a2, self.L1, self.m2, self.I2)

        #return (H_func, f_func)
    
        def right_hand_side(x, t, args):
            """Returns the derivatives of the states.

            Parameters
            ----------
               :x: ndarray (8,) [q, qd], the current state vector.
               :t: float, the current time.

            Returns
            -------
            dx : ndarray, shape(8,), yhe derivative of the state.
    
            """
            tau = controlLaw(x)   # The input forces 
            arguments = np.hstack((x, tau))      # States, input, and parameters
            dqd = np.array(solve(H_func(*arguments),  # Solving for the second derivatives
                                f_func(*arguments))).T[0]
            dq = x[4:]   # The first derivatives, i.e., the generalized velocities

            return np.hstack((dq, dqd))
        
        x0 = np.hstack((q0, qd0, np.zeros(2)))
        N = np.ceil(float(tmax)/dt)
        self.timeVector = np.linspace(0.0, tmax, num=N)  # Time vector
        x = odeint(right_hand_side, x0, self.timeVector)  # Numerical integration

        self.stateTrajectory = x[:,:4]
        qdots = x[:, 4:8]
        lams = x[:,8:]
        
        # Find the ground reaction forces
        grf = np.zeros((N, 2))
        for i in range(N):
            arguments = np.hstack((x[i], np.zeros(4)))
            Hf = H_func(*arguments)
            ff = f_func(*arguments)
            
        #self.grfTrajectory = grf

    def get_joints_and_endpoint(self):
        """ 
        Returns the current location of the two joints and the endpoint.
        """
        j0 = self.currentState[2:]
        j1 = j0 + self.L1*np.array([np.sin(self.currentState[0]), np.cos(self.currentState[0])])
        q1plusq2 = np.sum(self.currentState[:2])
        endp = j1 + self.L2*np.array([np.sin(q1plusq2), np.cos(q1plusq2)])

        return (j0, j1, endp)
        
    def get_CoM(self):
        """
        Returns the current center of mass of the complete pendulum model, as well as a list of 
        the center of mass of each of the links
        """

        (j0, j1, endp) = self.get_joints_and_endpoint()

        com1 = j0 + (j1-j0)*self.a1
        com2 = j1 + (endp-j1)*self.a2

        com = (self.m1*com1 + self.m2*com2)/(self.m1+self.m2)

        return (com, com1, com2)
    
    def get_next_state(self, t, dt):
        self.update_disturbance()
        self.integrator.update_state(dt)
        return self.integrator.odeint.y[:4]

    def get_next_input(self, t, dt):
        return -self.feedback(self.integrator.odeint.y[:4]-self.setpoint, t)



    def _reset_animation_fired(self):
        self.animation.reset()

    def _start_simulation_fired(self):
        # Make sure parameters are set
        self._update_parameters_fired()
        self.reset = True

        if self.delay <= 0.0:
            self.integrator = ScipyOde(self.ode)
        else:
            self.integrator = DelayIntegrator(\
                dps.get_cc_ode_fcn(self.a1, self.L1, self.m1, self.I1,
                                   self.a1, self.L1, self.m1, self.I1),
                self.feedback.ccode())


        # Simulate
        self.integrator.set_init_state(np.concatenate([self.x0, np.array([0.0,0])]))
        self.animation.init_dp(self.L1, self.L2, self.x0[0], self.x0[1])
        self.status = "Running simulation for %.4f seconds..." % self.tmax

        if hasattr(self, 'lqr'):
            QQ = self.lqr.Q
            RR = self.lqr.R
        else:
            QQ = np.eye(4)
            RR = np.eye(2)
            
        cost = self.animation.run_sim(self, self.tmax, 
                                      partial(self.quadratic_cost, 
                                              Q=QQ,
                                              R=RR))


        self.status = "Average control cost: %.4f" % cost


    def _feedbackType_changed(self, old, new):
        self._update_parameters_fired()
        if new == "LQR":
            self.lqr = LQR(self.A, self.B)
            self.lqr.configure_traits()
            self.stateFeedbackGain = self.lqr.compute_lqr()
            self._update_parameters_fired()

    def _update_parameters_fired(self):
        """ Called when parameters is to be updated """
        self.status = "Updating model parameters..."
        self.pendulum_ode = dps.get_ode_fcn(self.a1, self.L1, self.m1, self.I1, 
                                            self.a2, self.L2, self.m2, self.I2)
        self.disturbance = partial(self.disturbanceTorque, 
                                   np.array([[1,0],[0,1]]),
                                   0, self.noiseVar)
        
        self.set_linear_feedback(self.stateFeedbackGain)

        if self.reflexThreshold > 0:
            self.set_reflex_feedback(self.reflexThreshold, self.reflexAmplitude, 
                                     self.reflexDuration, self.reflexRefractoryPhase)

        if self.actuatorSaturation > 0:
            self.set_feedback_saturation((self.actuatorSaturation, self.actuatorSaturation))

        if self.stiffnessControl > 0:
            self.set_impedance_feedback(self.stiffnessControl)
            
        self.BB = np.eye(2)
        if self.actuate == "First only":
            self.BB[1,1] = 0.0

        if self.actuate == "Second only":
            self.BB[0,0] = 0.0

        self.A, Hinv = dps.get_linearized_ode(self.a1, self.L1, self.m1, self.I1, 
                                              self.a2, self.L2, self.m2, self.I2)

        self.B = np.dot(Hinv,self.BB)
        self.status = "Updating model parameters...Done!"


    def get_next_state(self, t, dt):
        self.update_disturbance()
        self.integrator.update_state(dt)
        return self.integrator.odeint.y[:4]

    def get_next_input(self, t, dt):
        return -self.feedback(self.integrator.odeint.y[:4]-self.setpoint, t)


    def get_set_point(self, t):
        return self.setpoint

    def quadratic_cost(self, x, u, Q, R):
        """ Function for computing a quadratic control cost """
        return ( np.dot(x, np.dot(Q, x)) + np.dot(u, np.dot(R, u)) )

    def update_disturbance(self):
        self.v = self.disturbance()

    def set_setpoint(self, xstar):
        self.setpoint = xstar

    def set_friction(self, f):
        """ Sets the friction factor. The friction provided must be a 2-element array
        which is multiplied with the angular velocity of the respective joints to provide a 
        frictional force at each joint.
        """
        self.friction = f


    def linear_feedback(self, x, t, K):
        return np.dot(K, np.array(x))

    def set_linear_feedback(self,K):
        self.feedback = partial(self.linear_feedback, K=K)

    def set_lqr_feedback(self, Q, R):
        K, S, E = control.lqr(self.A, self.BB, Q, R)
        print K
        self.set_linear_feedback(K)

    def set_swingup_feedback(self, umax):
        self.feedback = partial(self.swingup_feedback, umax=umax)

    def swingup_feedback(self, x, t, umax):
        
        th1dot = x[2]
        th2dot = x[3]

        return np.array([math.copysign(umax,th1dot), math.copysign(umax,th2dot)])

    def linear_feedback_deadzone(self, x, t, K, xlow, xhigh):
        xx = x.copy()
        for i in range(len(xx)):
            if x[i] < xhigh[i] and x[i]>xlow[i]:
                xx[i] = 0.0
        
        return np.dot(K,xx)

    def set_linear_feedback_deadzone(self, K, xlow, xhigh):
        self.feedback = partial(self.linear_feedback_deadzone, K=K, xlow=xlow, xhigh=xhigh)

    def feedback_saturation(self, x, t, originalFeedback, umax):
        u = originalFeedback(x,t)
        for i in range(len(u)):
            if u[i] > umax[i]:
                u[i] = umax[i]
            if u[i] < -umax[i]:
                u[i] = -umax[i]

        return u

    def set_feedback_saturation(self, umax):
        self.feedback = partial(self.feedback_saturation, originalFeedback=self.feedback, umax=umax)

    def add_feedback(self, x, t, newFeedback, originalFeedback):
        u1 = originalFeedback(x,t)
        u2 = newFeedback(x,t)
        return u1 + u2

    def set_reflex_feedback(self, thr, ampl, duration, refract):
        reflexFeedback = ReflexFeedback(thr, ampl, duration, refract, self.a1, self.a2, self.m1, self.m2)
        self.feedback = partial(self.add_feedback, newFeedback=reflexFeedback, originalFeedback=self.feedback)

    def set_impedance_feedback(self, stiffness):
        self.feedback = partial(self.impedance_feedback, stiffness=stiffness)

    def impedance_feedback(self, x, t, stiffness):
        """ Implements a simple elastic impedance feedback. """
        # Jacobian
        J = np.array([self.L1*self.cos(x[0]), self.L2*self.cos(x[1])])
        # horizontal position of end point
        L = self.L1*self.sin(x[0]) + self.L2*self.sin(x[1])
        return stiffness * L * J

    def disturbanceTorque(self, F, mu, sigma):
        v = np.array([DoublePendulum.rng.gauss(mu,sigma), DoublePendulum.rng.gauss(mu,sigma)])
        return np.dot(F,v)

    def linear_ode(self, t, x):
        """ Differential equations for linearized double pendulum with state feedback""" 

        u = -self.feedback(x[:4], t)
        self.u = u
        
        #2/0
        dx1 = np.dot(self.A, x[0:4]) + np.dot(self.BB, u + self.v)

        return [dx1[0], dx1[1], dx1[2], dx1[3], x[0]-self.setpoint[0], x[1]-self.setpoint[1]]

    def ode(self,t,x):
        """ Differential equations for double pendulum with state feedback""" 
        th1 = x[0]
        th2 = x[1]
        th1dt = x[2]
        th2dt = x[3]
        #e1 = x[4]
        #e2 = x[5]

        u = -self.feedback(x[0:4]-self.setpoint, t) - self.friction*x[2:4]
        self.u = u
        dx = self.pendulum_ode(x, np.dot(self.BB,u) + self.v)

        return [th1dt, th2dt, dx[0], dx[1], th1-self.setpoint[0], th2-self.setpoint[1]]
        #return [th1dt, th2dt, dx[0], dx[1]]


class ScipyOde:
    def __init__(self, odefcn):
        self.odeint = scipy.integrate.ode(odefcn).set_integrator('dopri')

    def run(self, dpmodel, x0, tmax, dt=0.01):
        self.odeint.set_initial_value(x0)
        while self.odeint.successful() and self.odeint.t < tmax:
            self.update_state(dt)

    def set_init_state(self, x0):
        self.odeint.set_initial_value(x0)

    def update_state(self, dt):
        self.odeint.integrate(self.odeint.t + dt)
        
class LQR(HasTraits):
    Q = Array(value=np.array([1,1, 1, 1]))
    R = Array(value=np.array([1., 1.]))

    compute_lqr = Button(label="Compute LQR")

    A = None
    B = None
    
    def __init__(self, A, B):
        udim = B.shape[1]
        self.R = np.diag(np.eye(udim))
        self.A = A
        self.B = B

    def compute_lqr(self):
        K, S, E = control.lqr(self.A, self.B, np.diag(self.Q), np.diag(self.R))
        return K

class ReflexFeedback():
    """ Function object which implements a reflex feedback. The feedback will be on for a given
    duration. After that, it will have a refractory phase, in which it can not be activated 
    again.
    The threshold should be given in m, and represents a horizontal displacement of the center
    of mass of the double pendulum. The position of the pendulum CoM is computed from the 
    joint angles and the distance from the joints to the respective CoMs of the two links.
    """
    def __init__(self, thr, amplitude, duration, refr, a1, a2, m1, m2):
        self.thr = thr
        self.amplitude = amplitude
        self.duration = duration
        self.refractoryduration = refr
        self.a1 = a1  # length from base to CoM of link 1
        self.a2 = a2  # length from joint to CoM of link 2
        self.m1 = m1  # mass of link 1
        self.m2 = m2  # mass of link 1
        self.invm = 1.0 / (m1+m2) # Inverse of the total mass. For faster computations
        self.firing = False
        self.armed = True
        self.output = np.array([0,0])
        self.stop = 0

    def __call__(self, x, t):
        if self.firing:
            if t > self.stop:
                self.firing = False
                self.armed = False
                self.stop = t + self.refractoryduration
                self.output[1] = 0.0
        else:
            if self.armed:
                if self.center_of_mass(x) > self.thr:
                    self.firing = True
                    self.output[1] = -self.amplitude
                    self.stop = t + self.duration
            else:
                # Check if refractory phase done
                if t > self.stop:
                    self.armed = True

        return self.output
    
    def center_of_mass(self, x):
        cm1 = self.a1*math.sin(x[0])
        cm2 = self.a2*math.sin(x[1])
        
        return (cm1*self.m1 + cm2*self.m2)*self.invm
    
def run_simulation():
    """ Entry point for this module. Will create a DP model, where the user may set the 
    traits of the model. Once the traits are set, an animation will be attached to the model, and the simulation started.
    """
    root = tk.Tk()
    width = 800
    height = 400
    frame_rate = 60

    dpanimation = DoublePendulumAnimation(root, width, height, frame_rate)
    dpmodel = DoublePendulum(dpanimation)
    #dpmodel.configure_traits(view=viewdp)
    dpmodel.configure_traits()

    

if __name__ == '__main__':
    #pendulum_model()
    #simulate_pendulum_test()
    #plt.show()
    run_simulation()
