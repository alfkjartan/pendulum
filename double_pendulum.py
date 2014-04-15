"""
Code for simulating a model of an actuated double pendulum 
"""
import os, sys
import random
from functools import partial
import pygame as pyg
from pygame.locals import *
import Tkinter as tk
from traits.api import HasTraits, Float, Button, Enum, String, Array
from traitsui.menu import Action
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from traitsui.api import *
import math
import numpy as np
import  scipy.integrate 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigCanvas, \
    NavigationToolbar2TkAgg as NavigationToolbar
import matplotlib.pyplot as plt
import double_pendulum_symbolic as dps
import control
from timeseriesplot import TimeSeriesPlot


class DoublePendulumAnimation:
    gravityDown = -1

    def __init__(self, master, w=600, h=400, fps=50.0):

        self.master = master
        self.fps = fps
        self.width = w
        self.height= h
        self.pygw = tk.Frame(master, height=h, width=w)
        #self.pygw.pack(fill=tk.X)
        self.pygw.pack()
        
        frame = tk.Frame(master)
        frame.pack()

        figframe = tk.Frame(master)
        figframe.pack(expand=True, fill=tk.BOTH)

        self.fig = Figure((3.0, 3.0), dpi=100)
        self.timeseriesplot = TimeSeriesPlot(self.fig)
        self.init_plot()
        self.canvas = FigCanvas(self.fig, master=figframe)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        #self.resetbutton = tk.Button(frame, text="Restart simulation", command=restart_sim)
        #self.resetbutton.pack()
        #self.quitbutton = tk.Button(frame, text="Quit", command=shutdown_app)
        #self.quitbutton.pack()

        os.environ['SDL_WINDOWID'] = str(self.pygw.winfo_id())

        master.update()

        self.window = pyg.display.set_mode((self.width,self.height))
        pyg.display.set_caption("Simulated double pendulum")

        self.screen = pyg.display.get_surface()
        pyg.init()
        

    def init_dp(self, length1, length2, th1, th2):
        """ Called by DP model when all parameters are set by the user, and the simulation 
        is about to start.
        """
        dplength = float(length1+length2)
        dplengthpxls = 0.45*self.height
        self.length1 = length1/dplength * dplengthpxls
        self.length2 = length2/dplength * dplengthpxls
        self.root_pos = (self.width/2, self.height/2)
        self.draw_pendulum(th1, th1+th2)


    def reset(self):
        self.fig.clear()
        self.init_plot()
        self.master.update()

    def init_plot(self):
        ts = []
        ts.append(dict(name='Inputs', 
                       timeseries=[ dict(label='u1', linewidth=1, color=(1,1,0)), 
                                    dict(label='u2', linewidth=1, color=(0.6,0.4,0))]))
        ts.append(dict(name='Joint angles', 
                       timeseries=[ dict(label='th1', linewidth=1, color=(0,1,1)), 
                                    dict(label='th2', linewidth=1, color=(0,0.4,0.6))]))
        #ts.append(dict(name='Power input', 
        #               timeseries=[ dict(label='power', linewidth=1, color=(1,0,1))]))
        ts.append(dict(name='Control cost', 
                       timeseries=[ dict(label='cost', linewidth=1, color=(1,0,1))]))
        
        self.timeseriesplot.init_plot(ts)
        
        #self.canvas.draw()
        
    

    def power(self, th1dot, th2dot, u1, u2):
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

    def draw_pendulum(self, th1, th2):

        point_radius = 5
        joint_color = (200,0,0)
        endp_color = (0, 100, 100)
        link_color = (100, 0, 100)
        link_width = 6
        
        joint_pos = (self.root_pos[0] - int(self.length1*math.sin(th1)), 
                     self.root_pos[1] - self.gravityDown*int(self.length1*math.cos(th1)))
        end_pos = (joint_pos[0] - int(self.length2*math.sin(th2)), 
                   joint_pos[1] - self.gravityDown*int(self.length2*math.cos(th2)))

        self.screen.fill((0,0,0))
        pyg.draw.line(self.screen, link_color, self.root_pos, joint_pos, link_width )
        pyg.draw.line(self.screen, link_color, joint_pos, end_pos, link_width )

        pyg.draw.circle(self.screen, joint_color, self.root_pos, point_radius, 0) 
        pyg.draw.circle(self.screen, joint_color, joint_pos, point_radius, 0)
        pyg.draw.circle(self.screen, endp_color, end_pos, point_radius, 0)
        
    def run_sim(self, dpmodel, tmax, costfunction):

        self.init_control_cost(costfunction)

        dt = 1.0 / self.fps
        clock = pyg.time.Clock()

        def check_events(events): 
            for event in events: 
                if event.type == QUIT: 
                    pyg.quit()
                    sys.exit(0) 
                    #   else: 
                    #      print event 

        pyg.display.update()

        dta = dict(th1=0, th2=0, u1=0, u2=0, cost=0)
        now = 0
        iterations = int(tmax/dt)
        for i in range(iterations):
            check_events(pyg.event.get()) 

            msElapsed = clock.tick(self.fps)
            #try:
            u = dpmodel.get_next_input(now, dt)
            th = dpmodel.get_next_state(now, dt)
            
            th0 = dpmodel.get_set_point(now)

            #except:
            #    break
            if np.isscalar(u):
                u1 = u
                u2 = 0
            else:
                u1 = u[0]
                if len(u) > 1:
                    u2 = dpmodel.u[1]
                else:
                    u2 = 0

            self.draw_pendulum(th[0], th[0]+th[1])
            pyg.display.update()
         
            if 0:
                now += dt

                dta['th1'] = th[0]
                dta['th2'] = th[1]
                dta['u1'] = u1
                dta['u2'] = u2
                dta['cost'] = self.update_control_cost(th-th0, u, dt)
                self.timeseriesplot.append_data(now, dta)

                self.master.update()
        return self.cost

def restart_sim():
    pass
def shutdown_app():
    pass

class Void (tk.Tk) :
    def __init__ (self, color='black') :
        tk.Tk.__init__(self)
        #self.wm_state('zoomed')
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (w, h))
        self.config(bg=color)
        self.overrideredirect(True)
        self.attributes('-topmost', True)

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
    g = 9.825
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

    def __init__(self, animation):
        self.animation = animation


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
