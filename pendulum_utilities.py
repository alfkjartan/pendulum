""" Utility functions for simulating and animating a multi-pendulum

    :Author: Kjartan Halvorsen
    :Date: 2016-04-22

"""
import numpy as np
import sympy as sy
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from matplotlib import animation

def animate_pendulum(t, qcoords, qsymb, subsdict, points, specialpoints, refFrame, filename=None):
    """Animates a chain of links and optionally saves it to file.

    Arguments
    ----------
        :t:        ndarray of shape (N,). Time array
        :qcoords:  ndarray of shape (N,n). Sequence of generalized coordinates
        :qsymb:    list of symbols, len(qsymb) = n, corresponding to the generalized coordinates  
        :subsdict: dictonary of symbols and values to substitute in the expression for the points.
        :points:   list 
                   List of endpoints and joints (sympy.physics.vector.point.Point). The order is important, since lines will be drawn between the sequence of
                   points. The points should only depend on the state, so all other symbols (parameters of the model) should be 
                   substituted with numerical values before calling animate_pendulum.
        :specialpoints:   list 
                   List of CoM and other special points (sympy.physics.vector.point.Point). The order is NOT important, since only markers will be drawn
        :refFrame: tuple (sympy.physics.vector.point.Point, sympy.physics.vector.frame.ReferenceFrame), The frame in which to express the movement, and the origin of this frame, which will be static.  

        :filename: string or None, optional
                   If true a movie file will be saved of the animation. This may take some time.

    Returns
    -------
        :fig:   matplotlib.Figure
                The figure.
        :anim:  matplotlib.FuncAnimation
                The animation.

    """

    (refpoint, frame) = refFrame
    
    # Define functions that will generate the x- and z-coordinates from the list of points
    x_func = sy.lambdify(qsymb, [me.dot(p.pos_from(refpoint).subs(subsdict), frame.x) for p in points])
    z_func = sy.lambdify(qsymb, [me.dot(p.pos_from(refpoint).subs(subsdict), frame.z) for p in points])
    x_func_s = sy.lambdify(qsymb, [me.dot(p.pos_from(refpoint).subs(subsdict), frame.x) for p in specialpoints])
    z_func_s = sy.lambdify(qsymb, [me.dot(p.pos_from(refpoint).subs(subsdict), frame.z) for p in specialpoints])

    #for p in points:
    #    x_funcs.append(sy.lambdify(q, me.dot(p, frame.x)))
    #    z_funcs.append(sy.lambdify(q, me.dot(p, frame.z)))
        
        
    # Set up the figure, the axis, and the plot elements we want to animate
    fig = plt.figure()
    
    # set the limits 
    axmin = -5
    axmax = 5
    
    # create the axes
    ax = plt.axes(xlim=(axmin, axmax), ylim=(axmin, axmax), aspect='equal', axisbg='black')
    
    # display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes, color=(0.5,0.9, 0.9))
    
    
    # blank line for the pendulum
    specialpoints_, = ax.plot([], [], 'o', markersize=10, color=(0.2, 0.8, 0.2))
    points_, = ax.plot([], [], 'o', markersize=6, color=(0.6, 0, 0.6))
    line_, = ax.plot([], [], lw=3, marker='o', markersize=8, color=(0.6, 0, 0))

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        line_.set_data([], [])
        points_.set_data([], [])
        specialpoints_.set_data([], [])
        return time_text, line_, points_, specialpoints_

    # animation function: update the objects
    def animate(i):
        time_text.set_text('time = {:2.2f} s'.format(t[i]))
        x = x_func(*qcoords[i, :])
        z = z_func(*qcoords[i, :])
        xs = x_func_s(*qcoords[i, :])
        zs = z_func_s(*qcoords[i, :])
        line_.set_data(x, z)
        points_.set_data(x, z)
        specialpoints_.set_data(xs, zs)
        return time_text,line_,points_, specialpoints_

    # call the animator function
    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
            interval=t[-1] / len(t) * 1000, blit=True, repeat=False)
    
    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=30, writer="avconv", codec='libx264')
        
    return (anim, fig)
