# /usr/bin/env python
import sys
import copy
from math import *
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import time
import tkinter as tk

# List of keywords for the input files
LATTICE_TAG = "box"
ATOM_TAG = "atom"
FORCE_TAG = "interactions"
TEMP_TAG = "temperature"
TIME_TAG = "time"
ANI_TAG = "recording"
CONST_TAG = "constraint"

X_TAG = "x"
Y_TAG = "y"
Z_TAG = "z"
VX_TAG = "vx"
VY_TAG = "vy"
VZ_TAG = "vz"
MASS_TAG = "mass"
NAME_TAG = "name"
INDEX_TAG = "index"
CONNECT_TAG = "connect"
T_TAG = "t"
TAU_TAG = "friction"
RS_TAG = "random-start"
BK_TAG = "box-k"
CK_TAG = "connect-k"
CR_TAG = "connect-r"
CM_TAG = "connect-range"
SIGMA_TAG = "sigma"
EPS_TAG = "epsilon"
RF_TAG = "random-f"
RN_TAG = "random-n"
DT_TAG = "dt"
RUNTIME_TAG = "total"
PLANE_TAG = "animate"
PLOT_TAG = "plot"
ANIDT_TAG = "anim-dt"
RECDT_TAG = "record-dt"
THERMAL_TAG = "thermalizing-t"
TYPE_TAG = "type"
VALUE_TAG = "value"
FREEZE_TAG = "freeze"
F_TAG = "force"
V_TAG = "velocity"

# Indices for selecting correct pieces of data
CK_INDEX = 0
CR_INDEX = 1
CM_INDEX = 2
SIGMA_INDEX = 3
EPS_INDEX = 4

T_INDEX = 0
TAU_INDEX = 1
RS_INDEX = 2

DT_INDEX = 0
RUNTIME_INDEX = 1

PLANE_INDEX = 0
ANIDT_INDEX = 1
RECDT_INDEX = 2
THERMAL_INDEX = 3
PLOT_INDEX = 4



def animate(particles, lattice_parameters, force_parameters):
    """
    Animates the simulation.
    
    Args:
        particles (list): list of :class:`Particle` objects
        lattice_parameters (list): list of lattice parameters
        margin (float): each axis will be drawn in the range [-margin , lattice parameter + margin]
    """
    # these set up the graphics window
    engine = tk.Tk()
    movie = Animation(engine, particles, lattice_parameters, force_parameters)
    engine.mainloop()
    
    

class Animation:
    """
    A class for creating an animation based on simulation results.
    
    The animation is drawn in a graphical window generated using tkinter.
    
    Args:
        root (tkinter.canvas): graphical engine object from tkinter
        particles (list): list of :class:`Particle` objects
        lattice_parameters (list): lattice parameters [Lx, Ly]
        force_parameters (list): interaction parameters
    """

    def __init__(self, root, particles, lattice_parameters, force_parameters):
        self.root = root
        self.root.title("Particle simulation")
        self.lattice_parameters = lattice_parameters
        
        max_L = np.max(lattice_parameters)
        self.SCALE = int(800/max_L) # scales graphics window size
        self.DT = 25 # time between animation frames in ms
        self.frame = 0 # frame counter
        self.n_frames = len(particles[0].trajectory) # number of frames
        
        Lx = lattice_parameters[0]
        Ly = lattice_parameters[1]
        self.r0 = force_parameters[CR_INDEX]
        self.maxr = force_parameters[CM_INDEX]
        self.maxr2 = force_parameters[CM_INDEX]**2

        # create the base for drawing graphics
        self.canvas = tk.Canvas(root, width=Lx*self.SCALE, height=Ly*self.SCALE)
        self.canvas.pack()
                          
        self.particles = particles

        # initialize lists for storing graphical representations
        # of particles and spring-like bonds
        self.atoms = [ ]
        self.bonds = [ ]
        
        # Loop over all particles and their bonds.
        # We create bonds before particles so that they are drawn
        # beneath the particles in the graphical representation.
        for p in self.particles:
        
            # coordinates of the particle
            x = p.trajectory[0][0]
            y = Ly-p.trajectory[0][1]
                
            # loop over all the bonds of this particle
            for c in p.connected_atoms:
            
                # prevent double counting
                if c.index > p.index:

                    # create a new graphical element (a line) to represent a bond
                    # between two particles
                    dummy1 = Particle(p.trajectory[0], 0, 0, "", -1, [])
                    dummy2 = Particle(c.trajectory[0], 0, 0, "", -1, [])
                    dr = dummy1.vector_to(dummy2, lattice_parameters)
                    new_bond = self.canvas.create_line(self.SCALE*x, self.SCALE*y, self.SCALE*(x+dr[0]), self.SCALE*(y-dr[1]), fill="white", width=0) 
                    self.bonds.append(new_bond)

                    # if the bond is short enough, draw it
                    rsq = dr @ dr 
                    if rsq < self.maxr2:        
                        w, c = self.get_bond_width_and_colour(np.sqrt(rsq))
                        self.canvas.itemconfig(new_bond, fill=c, width=w)

                    # if the bond is too long, it has broken and will not be drawn
                    else:    
                        self.canvas.itemconfig(new_bond, fill="white", width=0)
                        self.canvas.coords( new_bond, -10, -10, -10, -10 )

        # loop over all particles
        for p in self.particles:
        
            # coordinate, radius and color for the particle            
            x = p.trajectory[0][0]
            y = Ly-p.trajectory[0][1]
            r, c = self.get_particle_size_and_colour(p)
            
            # create a new graphical element (a circle) to represent a particle
            new_atom = self.canvas.create_oval(
                x*self.SCALE-r, y*self.SCALE-r, x*self.SCALE+r, y*self.SCALE+r, 
                fill=c, outline="black", width=3
                )
            self.atoms.append( new_atom )

        # give focus to the graphical window
        self.canvas.focus_set()
        
        print("\nanimating with...")
        print("n atoms: ",len(self.atoms))
        print("n bonds: ",len(self.bonds))
        
        # start a clock for handling the drawing of
        # new frames at a constant frequency
        self.clock()


    def rgb_to_hex(self, rgb):
        """
        Transforms RGB colour to hexadecimal representation.
            
        Args:
            rgb (vector): 3-component vector storing Red-Green-Blue values [0-255]

        Returns: 
            string: colour hex
        """
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


    def get_bond_width_and_colour(self, r):
        """
        Gives the width and colour of a bond.
        
        To visualize bond straining, we draw bonds that are close to
        breaking as thin red lines. This function calculates
        the proper width and colour for a bond of given length.
        
        Args:
            r (float): length of the bond
            
        Returns:
            float, string: line width, colour hex
        """
    
        # balanced: value at equilibrium
        # strained: value at breaking point
        balanced_c = np.array( [0,0,0] ) # black in rgb
        strained_c = np.array( [255,100,100] ) # red in rgb
        balanced_w = 3.0
        strained_w = 1.0
        
        # if r equals equilibrium bond length, ratio = 0
        # if r is at maximum bond length, ratio = 1
        ratio = np.min( [ (r-self.r0)**2 / (self.maxr-self.r0)**2, 1 ] )      

        # take linear averages of balanced and strained values
        rgb = strained_c*ratio + balanced_c*(1-ratio)
        width = strained_w*ratio + balanced_w*(1-ratio)
        
        return width, self.rgb_to_hex(rgb)
        

    def get_particle_size_and_colour(self, particle):
        """
        Gives the radius and colour of a particle.
        
        To make different particles more easily distinguishable,
        we draw them with various sizes and colours.

        Particles are distinguished according to their name.
        This function only knows the names used in this exercise.
                
        Args:
            particle (Particle): the particle
            
        Returns:
            float, string: circle radius, colour string
        """
        if particle.name == "load":
            return 0.8*self.SCALE, "black"
        elif particle.name == "fixed":
            return 0.3*self.SCALE, "gray"
        elif particle.name == "He":
            return 0.3*self.SCALE, "gray"
        elif particle.name == "O":
            return 0.35*self.SCALE, "white"
        else:
            return 0.2*self.SCALE, "red"
        

    def shift_particles(self,frame):
        """
        Updates the graphical representation.
        
        Shifts the graphical elements representing particles and bonds
        to the positions corresponding to the given frame.
        
        If the given frame number is too high,
        the animation simply loops back to beginning.
        
        Args:
            frame (int): number of the frame to be drawn
        """
        Lx = self.lattice_parameters[0]
        Ly = self.lattice_parameters[1]
            
        # loop over all particles
        c_index = 0
        for p, a in zip(self.particles, self.atoms):
        
            # save particle coordinates, radius and colour
            x = p.trajectory[frame][0]
            y = Ly-p.trajectory[frame][1]
            r, c = self.get_particle_size_and_colour(p)
            
            
            # if the simulation exploded, this may not work...    
            try: # if everything is ok
            
                # move the graphical representation of the particle
                # to the correct location
                self.canvas.coords(a,
                    x*self.SCALE-r, y*self.SCALE-r, x*self.SCALE+r, y*self.SCALE+r
                )
            
            except: # if something went wrong
            
                # jump back to beginning of animation
                self.frame = 0
                return
    
            
            # loop over all the bonds of this particle
            for c in p.connected_atoms:
            
                # prevent double counting
                if c.index > p.index:
                    dummy1 = Particle(p.trajectory[frame], 0, 0, "", -1, [])
                    dummy2 = Particle(c.trajectory[frame], 0, 0, "", -1, [])
                    dr = dummy1.vector_to(dummy2, self.lattice_parameters)

                    # the graphical representations of bonds are stored in
                    # the list self.bonds - we pick the correct one
                    b = self.bonds[c_index]
                    c_index += 1
        
                    # square of bond length
                    rsq = dr @ dr
                    
                    # if the bond is short enough, draw it
                    if rsq < self.maxr2:
                        w, c = self.get_bond_width_and_colour(np.sqrt(rsq))
                        self.canvas.itemconfig(b, fill=c, width=w)
                        self.canvas.coords( b, self.SCALE*x, self.SCALE*y, self.SCALE*(x+dr[0]), self.SCALE*(y-dr[1]) )

                    # if the bond is too long, it has broken and will not be drawn
                    else:    
                        self.canvas.itemconfig(b, fill="white", width=1)
                        self.canvas.coords(b, -10, -10, -10, -10 )


    def clock(self):
        """
        Updates the animation at a constant frequency.

        The graphical representation is updated to the next frame using 
        :meth:`Animation.shift_particles()`. This is repeated after a
        short delay to run the animation.
        """
        
        self.frame += 1
        self.shift_particles(self.frame % self.n_frames)
        
        self.canvas.after( self.DT , self.clock )



class Particle:
    """
    A class representing a point-like particle.
    
    Args:
        position (list): coordinates [x, y]
        velocity (list): components [vx, vy]
        mass (float): particle mass
        name (str): a name for the particle, to help distinguish it
        index (int): an identifying number
        connections (list): indices of the atoms that connect to this atom by a spring

    Attributes:
        position (list): coordinates [x, y]
        velocity (list): components [vx, vy]
        force (list): components [Fx, Fy]
        mass (float): particle mass
        name (str): a name for the particle, to help distinguish it
        index (int): an identifying number
        connected_atoms (list): list of :class:`Particle` objects connected to this atom
        constraint_type (str): one of FREEZE_TAG, F_TAG or V_TAG
        constraint_value (list): value of the force or velocity constraint  
    """

    def __init__(self, position, velocity, mass, name, index, connections ):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass
        self.name = name
        self.index = index
        self.connections = connections
        self.connected_atoms = []
        self.trajectory = []
        self.force = np.zeros(2)
        self.constraint_type = None
        self.constraint_value = 0
        
        
    def __str__(self):
        info =  "Particle "+self.name+"\n"
        info += "index = "+str(self.index)+"\n"
        info += "m = "+str(self.mass)+"\n"
        
        x = str( round( self.position[0], 2) )
        y = str( round( self.position[1], 2) )
        vx = str( round( self.velocity[0], 2) )
        vy = str( round( self.velocity[1], 2) )
        fx = str( round( self.force[0], 2) )
        fy = str( round( self.force[1], 2) )
        
        info += "[  x,  y ] = [ "+x+", "+y+" ]\n"
        info += "[ vx, vy ] = [ "+vx+", "+vy+" ]\n"
        info += "[ Fx, Fy ] = [ "+fx+", "+fy+" ]\n"
        if len(self.connected_atoms) > 0:
            for atom in self.connected_atoms:
                info += "connected to particle "+str(atom.index)+"\n"

        if self.constraint_type == FREEZE_TAG:
            info += "frozen\n"
        elif self.constraint_type == V_TAG:
            info += "constant velocity "+str(self.constraint_value)+"\n"
        elif self.constraint_type == F_TAG:
            info += "external force "+str(self.constraint_value)+"\n"

        return info
            
        
    def move(self, dt):
        """
        Set a new position for the particle as
        
        .. math::

            \\vec{r}(t+\\Delta t) = \\vec{r}(t) + \\vec{v} \Delta t + \\frac{1}{2m}\\vec{F} (\\Delta t)^2
             
        Args:
            dt (float): time step :math:`\\Delta t`
        """
        if self.constraint_type == FREEZE_TAG:
            pass
        else:
            self.position += self.velocity * dt + 0.5 * self.force/self.mass * dt*dt        
        
        
    def move_linearly(self, dt):
        """
        Set a new position for the particle as
        
        .. math::

            \\vec{r}(t+\\Delta t) = \\vec{r}(t) + \\vec{v} \Delta t
           
        Args:         
            dt (float): time step :math:`\\Delta t`
        """
        self.position += self.velocity * dt 
        
        
    def accelerate(self, dt, gamma=0):
        """
        Set a new velocity for the particle as
        
        .. math::

            \\vec{v}(t+\\Delta t) = \\vec{v}(t) + \\frac{1}{m}\\vec{F} \Delta t
           
        By default, the force :math:`F` is the total force
        applied on the particle by all other particles.
        It should be precalculated and stored in the
        attributes of the particle.
        
        If a non-zero gamma is given, a drag force
        :math:`\\vec{F}_\\text{drag} = - \\gamma m \\vec{v}`
        is also applied.
        
        If the particle is constrained, the constraints are also applied:
        
        * A frozen particle always has zero velocity.
        * A velocity constrained particle has constant velocity.
        * If an external force is applied, it is added to the total force.
           
        Args:         
            dt (float): time step :math:`\\Delta t`
            gamma (float): coefficient :math:`\\gamma` for the drag force
        """
        
        # check for constraints first
        if self.constraint_type == FREEZE_TAG: # static particle
            self.velocity = np.array([0.0, 0.0])
        elif self.constraint_type == V_TAG: # constant velocity
            self.velocity = self.constraint_value
        else:
        
            # apply acceleration due to force:
            # dv = a dt = F/m dt
            if self.constraint_type == F_TAG: # add external force
                dv = (self.constraint_value + self.force) * dt/self.mass
                
            else: # no constraints
                dv = self.force * dt/self.mass
        
            # Note for the Leapfrog algorithm:
            #
            # If gamma = 0, update simply with v(i+1/2) = v(i-1/2) + dv.
            #
            # If gamma > 0, one must solve for the new velocity v(i+1/2) from
            # v(i+1/2) = v(i-1/2) + dv - gamma [ v(i+1/2) + v(i-1/2) ]/2 dt.
            #
            self.velocity = ( (1 - 0.5*gamma*dt)*self.velocity + dv) / (1 + 0.5*gamma*dt)
        
        
    def save_position(self):
        """
        Save the current position of the particle.
        
        Note: in a real large-scale simulation one would
        never save trajectories in memory. Instead, these
        would be written to a file for later analysis.
        """
        self.trajectory.append( [ self.position[0], self.position[1] ] )
        
        
    def kinetic_energy(self):
        """
        Calculates the kinetic energy of the particle.
        
        Returns: 
            float: kinetic energy
        """
        return 0.5 * self.mass * (self.velocity @ self.velocity)
        
    
    def wrap(self, lattice):
        """
        If the particle is outside of the simulation area,
        its position is shifted by a suitable multiple of lattice
        vectors so that the particle ends up back inside the simulation area.
        
        Args:
            lattice (list): lattice parameters [Lx, Ly]
        """
        
        for i in range(2):
            if self.position[i] < 0:
                multi = -self.position[i] // lattice[i] + 1
                self.position[i] += multi*lattice[i]
            if self.position[i] > lattice[i]:
                multi = self.position[i] // lattice[i]
                self.position[i] -= multi*lattice[i]  
    
            
    def vector_to(self, other_particle, lattice):
        """
        Returns the vector pointing from the position of
        this particle to the position of other_particle.
        
        Takes periodic boundary conditions into account.
        
        Args:
            other_particle (Particle): the end point of the vector
            lattice (list): lattice parameters [Lx, Ly]
            
        Returns:
            array: vector pointing from this to the other particle
        """
        
        vector_to = other_particle.position - self.position
        
        for i in range(2):            
            if vector_to[i] < -lattice[i]/2:
                multi = (-vector_to[i] - lattice[i]/2) // lattice[i] + 1
                vector_to[i] += multi*lattice[i]
            elif vector_to[i] > lattice[i]/2:
                multi = (vector_to[i] - lattice[i]/2) // lattice[i] + 1
                vector_to[i] -= multi*lattice[i]
        
        return vector_to
    
    
    def distance_squared_to(self, other_particle, lattice):
        """
        Calculates and returns the square of the 
        distance between this and another particle using :meth:`vector_to`.
        
        Args:
            other_particle (Particle): the end point of the vector
            lattice (list): lattice parameters [Lx, Ly]
            
        Returns:
            float: squared distance from this to the other particle
        """
        vec = self.vector_to(other_particle, lattice)

        return vec @ vec


    def distance_to(self, other_particle, lattice):
        """
        Calculates and returns the distance between this
        and another particle using :meth:`vector_to`.
        
        Args:
            other_particle (Particle): the end point of the vector
            lattice (list): lattice parameters [Lx, Ly]
            
        Returns:
            float: distance from this to the other particle
        """
        vec = self.vector_to(other_particle,lattice)

        return sqrt( vec @ vec )


    def unit_vector_to(self, other_particle, lattice):
        """
        Returns the unit vector pointing from the position of
        this particle towards the position of other_particle using :meth:`vector_to`.
        
        Args:
            other_particle (Particle): the end point of the vector
            lattice (list): lattice parameters [Lx, Ly]
            
        Returns:
            array: unit vector pointing from this to the other particle
        """
        vec = self.vector_to(other_particle, lattice) 
        return vec / sqrt( vec @ vec )
        
       
        


def find_info(lines, tag):
    """
    Searches for the information wrapped in the given tag
    among the given lines of text.
    
    If tag is, e.g., "foo", the function searches for the start tag
    <foo> and the end tag </foo> and returns the lines of information
    between them.
    
    The function only finds the first instance of the given tag.
    However, in order to catch multiple instances of the tag, the
    function also returns all the information in lines following
    the end tag.
    
    For instance, if lines contains the strings:

    .. code-block ::
    
        aa
    
        <foo>
        bb
        cc
        </foo>
    
        dd
        ee
        ff
    
    the function will return two lists: ["bb", "cc"], ["", "dd", "ee", "ff"].
    
    Args:
        lines (list): the information as a list of strings
        tag (str): the tag to search
    
    Returns: 
        list, list: the lines between start and end tags, the lines following the end tag
    """
    info = []
    is_relevant = False
    line_number = 0
        
    # go through the data
    for i in range(len(lines)):
        line = lines[i]
        
        if is_relevant: # if we have found the starting tag, record information 
            info.append(line)
            
        contents = line.strip() # remove whitespace at the start and end of the line
        
        if len(contents) > 0: # skip empty lines
        
            if contents[0] == "<" and contents[-1] == ">": # is this a tag?
            
                if contents[1:-1] == tag: # found the starting tag

                    if not is_relevant: # we had not yet found the tag
                        is_relevant = True # the following lines are relevant
                        line_number = i
                        
                    else: # we had already started this tag
                        print("Found tag <"+tag+"> while already reading <"+tag+">")
                        raise Exception("parsing error")
                        
                if contents[1:-1] == "/"+tag: # found the end tag
                    return info, lines[i+1:]
    
        
    # we end up here, if we reach the end of the file
    
    if is_relevant: # the file ends while reading info (start tag was found, but no end tag)
        print("Reached the end of file while parsing <"+tag+"> from line "+str(line_number+1))
        raise Exception("parsing error")
        
    elif info == []: # the tag was not found
        #print("Tag <"+tag+"> was not found")
        return [], lines
        

        
def parse_line(line):
    """
    Separates tag and info on a line of text.
    
    The function also removes extra whitespace and comments separated with #.
    
    For instance if line is " x :  1.23  # the x coordinate",
    the function returns ("x", "1.23").
    
    Args:
        line (str): a string of information
    
    Returns: 
        str, str: tag, info
    """

    parts = line.split(":")
    tag = ""
    info = ""
    
    if len(parts) > 1:
        tag = parts[0].strip()
        info = parts[1].split("#")[0].strip()
        
    return tag, info
    

def read_box(lines, default=10.0):
    """
    Reads lattice parameter info from given lines.
    
    Args:
        lines (list): information as a list of strings
        default (float): the default lattice parameter in all directions
    
    Returns: 
        list: lattice parameters [Lx, Ly]
    """
    lattice = [default]*2
    
    for line in lines:
        tag, info = parse_line(line)
        if tag == X_TAG:
            lattice[0] = float(info)
        elif tag == Y_TAG:
            lattice[1] = float(info)

    return lattice
    

def read_atom(lines):
    """
    Reads the properties of a single particle.
    
    Args:
        lines (list): information as a list of strings
    
    Returns: 
        Particle: a new Particle object created from the given information
    """
    
    i = 0
    n = "X"
    m = 1.0
    c = []
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0
    
    for line in lines:
        tag, info = parse_line(line)
        if tag == X_TAG:
            x = float(info)
        elif tag == Y_TAG:
            y = float(info)
        elif tag == VX_TAG:
            vx = float(info)
        elif tag == VY_TAG:
            vy = float(info)
        elif tag == NAME_TAG:
            n = info
        elif tag == MASS_TAG:
            m = float(info)
        elif tag == INDEX_TAG:
            i = int(info)
        elif tag == CONNECT_TAG:
            c.append( int(info) )
    
    return Particle(position=[x,y], velocity=[vx,vy], 
                    index=i, name=n, mass=m, connections=c)
                        


def read_temperature(lines):
    """
    Reads temperature parameters.
    
    The information is returned as a list with these elements:

    * params[T_INDEX] : external temperature
    * params[TAU_INDEX] : thermostat strength (0 for no thermostat)
    * params[RS_INDEX] : random start switch: If "yes", all atoms will be given new
      random velocities following the Maxwell-Boltzmann distribution
      at the start of the simulation.
    
    Args:
        lines (list): information as a list of strings
    
    Returns: 
        list: temperature parameters
    """

    t_params = [0]*3
    
    for line in lines:
        tag, info = parse_line(line)
        if tag == T_TAG:
            t_params[T_INDEX] = float(info)
        elif tag == TAU_TAG:
            t_params[TAU_INDEX] = float(info)
        elif tag == RS_TAG:
            t_params[RS_INDEX] = info
    
    return t_params
    
    
def read_interactions(lines):
    """
    Reads interaction parameters.
    
    The information is returned as a list with these elements:

    * params[CK_INDEX] : spring constant
    * params[CR_INDEX] : spring equilibrium length
    * params[CM_INDEX] : spring maximum length (no force beyond this separation)
    * params[SIGMA_INDEX] : Lennard-Jones sigma parameter
    * params[EPS_INDEX] : Lennard-Jones epsilon parameter
    
    Args:
        lines (list): information as a list of strings    

    Returns: 
        list: interaction parameters
    """

    f_params = [0]*5
    
    for line in lines:
        tag, info = parse_line(line)
        
        if tag == CK_TAG:
            f_params[CK_INDEX] = float(info)
        elif tag == CR_TAG:
            f_params[CR_INDEX] = float(info)
        elif tag == CM_TAG:
            f_params[CM_INDEX] = float(info)
        elif tag == SIGMA_TAG:
            f_params[SIGMA_INDEX] = float(info)
        elif tag == EPS_TAG:
            f_params[EPS_INDEX] = float(info)
    
    return f_params
    
    
def read_constraint(lines):
    """
    Reads one constraint.
    
    Args:
        lines (list): information as a list of strings
        
    Returns: 
        str, str, float: constraint info as (name, type, value)
    """

    type = ""
    name = ""
    value = 0
    
    for line in lines:
        
        tag, info = parse_line(line)
        if tag == TYPE_TAG:
            type = info
        elif tag == NAME_TAG:
            name = info
        elif tag == VALUE_TAG:
            vals = info[1:-1].split(",")
            value = []
            for v in vals:
                value.append(float(v))
                        
            value = np.array(value)
            
    return name, type, value
    
    
def read_timing(lines):
    """
    Reads timing parameters.
    
    The information is returned as a list with these elements:

    * params[DT_INDEX] : simulation time step
    * params[RUNTIME_INDEX] : total simulation time
    
    Args:
        lines (list): information as a list of strings
    
    Returns: 
        list: timing parameters
    """

    t_params = [0.1, 0.1]
    
    for line in lines:
        tag, info = parse_line(line)
        if tag == DT_TAG:
            t_params[DT_INDEX] = float(info)
        elif tag == RUNTIME_TAG:
            t_params[RUNTIME_INDEX] = float(info)
    
    return t_params
    
    
def read_recording(lines):
    """
    Reads data recording parameters.
    
    The information is returned as a list with these elements:

    * params[PLANE_INDEX] : animation switch: If "yes", an animation will be drawn at the end
    * params[PLOT_INDEX] : plot switch: If "yes", temperature and pressure will be plotted at the end
    * params[ANIDT_INDEX] : simulation time between animation frames
    * params[RECDT_INDEX] : simulation time between recording of physical data
    * params[THERMAL_INDEX] : simulation time before data collecting begins
    
    Args:
        lines (list): information as a list of strings
        
    Returns: 
        list: timing parameters
    """

    a_params = ['no', 1.0, 1.0, 0.0, 'no']
    
    for line in lines:
        tag, info = parse_line(line)
        if tag == PLANE_TAG:
            a_params[PLANE_INDEX] = info
        elif tag == ANIDT_TAG:
            a_params[ANIDT_INDEX] = float(info)
        elif tag == RECDT_TAG:
            a_params[RECDT_INDEX] = float(info)
        elif tag == THERMAL_TAG:
            a_params[THERMAL_INDEX] = float(info)
        elif tag == PLOT_TAG:
            a_params[PLOT_INDEX] = info
    
    return a_params
    

def read_system(filename):
    """
    Read particle and lattice data from a file.
    
    Args:
        filename (str): the file containing system information
    
    Returns: 
        list, list: lattice parameters, list of :class:`Particle` objects
    """
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    lattice_info, dummy = find_info( lines, LATTICE_TAG )
    lattice_parameters = read_box( lattice_info )
    
    particles = []
    success = True
    
    part_lines = lines
    
    while success:
        atom_info, part_lines = find_info( part_lines, ATOM_TAG )
        if len(atom_info) > 0:
            atom = read_atom( atom_info )
            particles.append(atom)
        else:
            success = False
    
    # add connections to Particle objects
    find_connected_atoms(particles)
    
    return lattice_parameters, particles
    

def read_physics(filename):
    """
    Read interaction and temperature data from a file.
    
    Args:
        filename (str): the file containing physical information
    
    Returns: 
        list, list: interaction parameters, temperature parameters
    """
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    force_info, dummy = find_info( lines, FORCE_TAG )
    interaction_parameters = read_interactions( force_info )

    t_info, dummy = find_info( lines, TEMP_TAG )
    temperature_parameters = read_temperature( t_info )        
    
    return interaction_parameters, temperature_parameters
    
    
def read_constraints(filename):
    """
    Read constraint settings from a file.
    
    Args:
        filename (str): the file containing cosntraint information
    
    Returns: 
        list: list of constraints
    """

    f = open(filename)
    lines = f.readlines()
    f.close()
    
    constraints = []
    success = True
    
    part_lines = lines
    
    while success:
        c_info, part_lines = find_info( part_lines, CONST_TAG )
        if len(c_info) > 0:
            name, type, value = read_constraint( c_info )

            constraints.append([name, type, value])

        else:
            success = False
            
    return constraints
    
    
def read_timescale(filename):
    """
    Read simulation and recording timing data from a file.
    
    Args:
        filename (str): the file containing simulation information
    
    Returns: 
        list, list: simulation timescale parameters, data recording parameters
    """
    
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    time_info, dummy = find_info( lines, TIME_TAG )
    time_parameters = read_timing( time_info )
    
    rec_info, dummy = find_info( lines, ANI_TAG )
    rec_parameters = read_recording( rec_info )
    
    return time_parameters, rec_parameters
      
    
def find_connected_atoms(particles):
    """
    Builds connections between :class:`Particle` objects.
    
    Particles can be defined with a connection to another particle,
    which means that there is a spring-like bond connecting these two particles.
    As particle data is read, only the indices of connected particles are read
    and saved in the :class:`Particle` objects. This function goes through all Particles
    and adds links to the Particle objects that are connected.
    
    After this operation, each Particle has a list named connected_atoms
    containing the other Particle objects that are connected to it.
    
    Connections are always reciprocal: If A is connected to B, then B is connected to A.
    The original particle data needs not fulfill this condition, but the function
    will always form connections to both connected Particles even if only one of them
    orginally declared the connection.
    
    Args:
        particles (list): list of :class:`Particle` objects
    """

    # remove duplicate indices
    for atom in particles:
        atom.connections = list( dict.fromkeys(atom.connections) )
                
    # find the atoms whose indices are in the list of connections
    for atom_A in particles: # go through all particles
        for index in atom_A.connections: # go through all connected indices
            
            for atom_B in particles: # go through all other atoms
            
                # are we looking for atom B?
                if atom_B.index == index and atom_B.index != atom_A.index:
                    
                    # if atoms A and B are not already connected, connect
                    if atom_B not in atom_A.connected_atoms:
                        atom_A.connected_atoms.append(atom_B)
                        atom_B.connected_atoms.append(atom_A)
                    
                    # If A is connected to B, B must be connected to A.
                    # If the reciprocal connection is missing, add it.
                    if not atom_A.index in atom_B.connections:
                        atom_B.connections.append(atom_A.index)
                    

      
def apply_constraints(particles, constraints):
    """
    Saves constraint information in :class:`Particle` objects.
    
    Args:
        particles (list): list of :class:`Particle` objects
        constraints (list): list of constraints
    """

    # go through all constraints
    for c in constraints:
    
        name = c[0]
        type = c[1]
        value = c[2]
        
        # go through all particles
        for atom in particles:
        
            # apply the constraint iff the name of the atom 
            # matches the name of the constraint
            if atom.name == name:
            
                atom.constraint_type = type
                atom.constraint_value = value
                
                    
 
def spring_energy(atom_A, atom_B, k, r0, rmax, lattice_parameters):
    """
    Calculate the spring potential energy of two connected atoms.

    Denote the distance between the atoms as :math:`r` and
    the maximum spring length as :math:`r_\\max`.

    If the atoms are close enough, :math:`r < r_\\max`, the energy is

    .. math ::
         U = \\frac{1}{2} k (r - r_0)^2,
        
    where :math:`k` is the spring constant and :math:`r_0` is the
    equilibrium distance.
    
    If the atoms are too far apart, :math:`r \ge r_\\max`, the energy is
    
    .. math ::
         U = \\frac{1}{2} k (r_\\max - r_0)^2.
    
    With these definitions, the potential energy has the minimum
    :math:`U(r_0) = 0` and increases parabolically. 
    At large separations, :math:`r > r_\\max`, the energy is constant.

    The function does not check if the particles should interact.
    It assumes they always do.
    
    Args:
        atom_A (Particle): atom taking part in the interaction
        atom_B (Particle): atom taking part in the interaction
        k (float): spring constant :math:`k`
        r0 (float): spring equilibrium length :math:`r_0`
        rmax (float): maximum spring length :math:`r_\\max`
        lattice_parameters (list): lattice parameters [Lx, Ly]
    
    Returns: 
        float: potential energy :math:`U`
    """
    
    dist_AB = atom_A.distance_to(atom_B, lattice_parameters)
    
    if dist_AB < rmax:
        u = 0.5 * k * (dist_AB - r0)**2
    else:
        u = 0.5 * k * (rmax - r0)**2
    
    return u
 
    
def spring_force(atom_A, atom_B, k, r0, rmax, lattice_parameters):
    """
    Calculate the spring force that atom B applies on atom A.

    Returns the force associated with the potential energy 
    given by the function :meth:`spring_energy`:
    
    .. math::
    
        \\vec{F}_A = - \\nabla_A U,
        
    where the energy is differentiated with respect to the coordinates of atom A.
    
    The function does not check if the particles should interact.
    It assumes they always do.
    
    Args:
        atom_A (Particle): atom taking part in the interaction
        atom_B (Particle): atom taking part in the interaction
        k (float): spring constant :math:`k`
        r0 (float): spring equilibrium length :math:`r_0`
        rmax (float): maximum spring length :math:`r_\\max`
        lattice_parameters (list): lattice parameters [Lx, Ly]
    
    Returns: 
        array: force acting on atom A, [Fx, Fy]
    """
    
    vec_AB = atom_A.vector_to(atom_B, lattice_parameters)
    dist_AB = sqrt( vec_AB @ vec_AB )
    
    if dist_AB < rmax:
        force_to_A = k * (dist_AB - r0) * vec_AB/dist_AB
    else:
        force_to_A = np.zeros(2)
    
    return force_to_A
    
    
def lj_energy(atom_A, atom_B, sigma_sixth, epsilon, lattice_parameters):
    """
    Calculate the Lennard-Jones potential energy of two atoms.

    Denote the distance between the atoms as :math:`r`.
    The potential energy is calculated as:
    
    .. math ::
        U = 4 \\epsilon \\left( \\frac{ \\sigma^{12} }{ r^{12} } 
        - \\frac{ \\sigma^6 }{ r^6 } \\right),
    
    where :math:`\\sigma` and :math:`\\epsilon` are parameters of the model.
    
    Args:
        atom_A (Particle): atom taking part in the interaction
        atom_B (Particle): atom taking part in the interaction
        sigma_sixth (float): parameter :math:`\\sigma^6`
        epsilon (float): parameter :math:`\\epsilon`
        lattice_parameters (list): lattice parameters [Lx, Ly]
    
    Returns: 
        float: potential energy :math:`U`
    """

    dist_2 = atom_A.distance_squared_to(atom_B, lattice_parameters)

    dist_6 = dist_2 * dist_2 * dist_2
    dist_12 = dist_6 * dist_6

    u = 4.0 * epsilon * ( sigma_sixth*sigma_sixth/dist_12 - sigma_sixth/dist_6)
    
    return u
    
    
def lj_force(atom_A, atom_B, sigma_sixth, epsilon, lattice_parameters):
    """
    Calculate the Lennard-Jones force that atom B applies on atom A.

    Returns the force associated with the potential energy U 
    given by the function :meth:`lj_energy`:
    
    .. math::
    
        \\vec{F}_A = - \\nabla_A U,
        
    where the energy is differentiated with respect to the coordinates of atom A.
    
    Args:
        atom_A (Particle): atom taking part in the interaction
        atom_B (Particle): atom taking part in the interaction
        sigma_sixth (float): parameter :math:`\\sigma^6`
        epsilon (float): parameter :math:`\\epsilon`
        lattice_parameters (list): lattice parameters [Lx, Ly]
    
    Returns: 
        array: force acting on atom A, [Fx, Fy]
    """

    vec_AB = atom_A.vector_to(atom_B, lattice_parameters)
    dist_2 = vec_AB @ vec_AB

    dist_6 = dist_2 * dist_2 * dist_2
    dist_12 = dist_6 * dist_6
                   
    force_to_A = - 4.0 * epsilon * \
                (12.0 * sigma_sixth*sigma_sixth / dist_12 - 6.0 * sigma_sixth / dist_6 ) \
                * vec_AB / dist_2
        
    return force_to_A
    


    
    
def calculate_forces(particles, force_parameters, lattice_parameters):
    """
    Calculate the total force acting on every atom.
    
    Saves the result to each :class:`Particle` object in particle.force.
    
    Args:
        particles (list): list of :class:`Particle` objects
        force_parameters (list): interaction parameters
        lattice_parameters (list): lattice parameters [Lx, Ly]

    Returns:
        float: the virial
    """
    
    connection_k = force_parameters[CK_INDEX] # spring constant for bonds
    connection_r = force_parameters[CR_INDEX] # equilibrium distance for bonds
    connection_rmax = force_parameters[CM_INDEX] # max distance for bonds
    sigma = force_parameters[SIGMA_INDEX] # sigma for Lennard-Jones potential
    epsilon = force_parameters[EPS_INDEX] # epsilon for Lennard-Jones potential    
    sigma_6 = sigma**6
    
    virial = 0.0
    
    # first loop: reset forces
    for atom_i in particles:
        
        # reset all forces
        atom_i.force = np.zeros(2)
        
    # second loop: add spring and Lennard-Jones forces
    for i in range(len(particles)):
    
        atom_i = particles[i]
        
        # apply spring force only if the particles are connected
        for atom_j in atom_i.connected_atoms:
            
            # prevent double counting
            if atom_i.index > atom_j.index:
                
                # calculate force on atom i
                force_to_i = spring_force(atom_i, atom_j, 
                    connection_k, connection_r, connection_rmax, lattice_parameters)
                    
                atom_i.force += force_to_i
            
                # law of action and reaction: opposite force on atom j
                atom_j.force -= force_to_i
                
                virial -= atom_i.vector_to(atom_j, lattice_parameters) @ force_to_i
    
        if sigma > 0:
            # apply LJ force for particles that are not spring-connected
            for j in range(0,i): # only cases j < i to prevent double counting
        
                atom_j = particles[j]
                
                if atom_j.index not in atom_i.connections:
        
                    # calculate force on atom i
                    force_to_i = lj_force(atom_i, atom_j, sigma_6, epsilon, lattice_parameters)
            
                    atom_i.force += force_to_i
            
                    # law of action and reaction: opposite force on atom j
                    atom_j.force -= force_to_i
              
                    virial -= atom_i.vector_to(atom_j, lattice_parameters) @ force_to_i
                
    return virial



def calculate_spring_potential_energy(particles, force_parameters, lattice_parameters):
    """
    Calculate the total potential energy of all spring-like bonds.
    
    Args:
        particles (list): list of :class:`Particle` objects
        force_parameters (list): interaction parameters
        lattice_parameters (list): lattice parameters [Lx, Ly]
    
    Returns: 
        float: total spring potential energy
    """
    
    connection_k = force_parameters[CK_INDEX] # spring constant for bonds
    connection_r = force_parameters[CR_INDEX] # equilibrium distance for bonds
    connection_rmax = force_parameters[CM_INDEX] # max distance for bonds

    u = 0.0

    for atom_i in particles:
            
        # calculate spring energy only if the particles are connected
        for atom_j in atom_i.connected_atoms:
            
            # prevent double counting
            if atom_i.index > atom_j.index:
            
                # calculate force on atom i
                u += spring_energy(atom_i, atom_j, 
                    connection_k, connection_r, connection_rmax, lattice_parameters)
                
    return u


def calculate_lj_potential_energy(particles, force_parameters, lattice_parameters):
    """
    Calculate the total potential energy of all Lennard-Jones interactions.
    
    Args:
        particles (list): list of :class:`Particle` objects
        force_parameters (list): interaction parameters
        lattice_parameters (list): lattice parameters [Lx, Ly]
    
    Returns: 
        float: total LJ potential energy
    """

    sigma = force_parameters[SIGMA_INDEX] # sigma for Lennard-Jones potential
    epsilon = force_parameters[EPS_INDEX] # epsilon for Lennard-Jones potential    
    sigma_6 = sigma**6

    u = 0.0
    
    for i in range(len(particles)):
            
        atom_i = particles[i]
            
        # calculate LJ energy for all particle pairs
        for j in range(0,i): # only cases j < i to prevent double counting
        
            atom_j = particles[j]

            if atom_j.index not in atom_i.connections:
        
                # calculate force on atom i
                u += lj_energy(atom_i, atom_j, sigma_6, epsilon, lattice_parameters)

    return u
    
           

def calculate_momentum(particles):
    """
    Calculate the total momentum of the system.
    
    Args:
        particles (list): list of :class:`Particle` objects
    
    Returns: 
        array: momentum [px, py]
    """
    p = np.array( [0.0, 0.0] )
    for atom in particles:
        p += atom.mass * atom.velocity
        
    return p
    
    
def calculate_kinetic_energy(particles):
    """
    Calculate the total kinetic energy of the system.
    
    Args:
        particles (list): list of :class:`Particle` objects
    
    Returns: 
        float: total kinetic energy
    """
    k = 0.0
    for atom in particles:
        k += atom.kinetic_energy()
    
    return k
    
           
      
def calculate_temperature(particles):
    """
    Calculate the current instantaneous temperature.
    
    The calculation is based on the kinetic energies of particles.
    
    According to the equipartition principle, each quadratic
    degree of freedom (DOF) stores, on average, the energy
    
    .. math ::
        \\langle E_\\text{DOF} \\rangle = \\frac{1}{2} k_B T,
    
    where :math:`k_B` is the Boltzmann constant and :math:`T` is the temperature.
    
    In 2D, every unconstrained atom has 2 degrees of freedom for their 
    linear movement, so the total kinetic energy of the system is,
    on average,
    
    .. math ::
        K = 2 N_\\text{atoms} \\langle E_\\text{DOF} \\rangle  = N_\\text{atoms} k_B T.
    
    For simplicity, we set :math:`k_B = 1`, so the temperature is
    
    .. math::

        T = \\frac{1}{ N_\\text{atoms}} K.
    
    At the macroscopic scale, the kinetic energy of a microscopic system may be 
    observed either as kinetic energy (movement) or internal energy (hotness).
    This function assumes there is no macroscopic kinetic energy so that
    all microscopic kinetic energy can be interpreted as internal energy.
    That is, this function assumes the system as a whole is at rest and the
    movement of particles is random.
    
    If the system has a moving center of mass or rotation around a center, 
    there is collective motion which is observed as macroscopic movement.
    In such a case this function systematically reports too high temperatures,
    because not all of the microscopic energy is internal energy at the
    macro scale.
    
    Args:
        particles (list): list of :class:`Particle` objects
    
    Returns: 
        float: temperature
    """
    
    dof = 0
    
    # go through all atoms
    for atom in particles:
    
        # ignore constrained atoms - they have no degrees of freedom
        if atom.constraint_type == FREEZE_TAG:
            dof += 0
        elif atom.constraint_type == V_TAG:
            dof += 0
        else:
            dof += 2
    
    return 2.0/dof*calculate_kinetic_energy(particles)
     
  
def calculate_pressure(particles, lattice_parameters, virial):
    """
    Calculate the current pressure.
    
    For a molecular simulation with constant pressure, volume and temperature, 
    one can derive the relation
    
    .. math::
    
       pV = Nk_B T + \\frac{1}{d} \\langle \\sum_i \\vec{F}_i \\cdot \\vec{r}_i \\rangle,
       
    where :math:`p, V, N, k_B, T, d, \\vec{F}_i, \\vec{r}_i` are, respectively,
    pressure, volume, number of atoms, Boltzmann constant, temperature,
    number of dimensions, force acting on atom :math:`i` and position of atom :math:`i`.
    The sum of the products of forces and positions is known as the virial
    and it should be calculated with :meth:`calculate_forces`.
    
    The function uses this relation to solve the effective instantaneous pressure as
    
    .. math ::

       p = \\frac{1}{V} Nk_B T + \\frac{1}{dV} \\sum_i \\vec{F}_i \\cdot \\vec{r}_i.

    This is not necessarily the true instantaneous pressure, but calculating
    the average of this quantity over an extended simulation should converge
    towards the true pressure.
    
    Args:
        particles (list): list of :class:`Particle` objects
        lattice_parameters (list): lattice parameters [Lx, Ly]
        virial (float): the virial
    
    Returns: 
        float: pressure
    """
    volume = lattice_parameters[0]*lattice_parameters[1]
    n = len(particles)
    t = calculate_temperature(particles)
        
    return (n*t + virial/2) / volume  
    

def update_positions(particles, dt):
    """
    Update the positions of all particles according to
    
    .. math::
        \\Delta \\vec{r} = \\vec{v} \\Delta t + \\frac{1}{2m} \\vec{F} (\\Delta t)^2

    using :meth:`Particle.move`.

    Args:
        particles (list): list of :class:`Particle` objects
        dt (float): time step :math:`\\Delta t`
    """
    for atom in particles:
        atom.move(dt)



def update_positions_without_force(particles, dt):
    """
    Update the positions of all particles according to
    
    .. math::
        \\Delta \\vec{r} = \\vec{v} \\Delta t

    using :meth:`Particle.move_linearly`.
    
    Args:
        particles (list): list of :class:`Particle` objects
        dt (float): time step :math:`\\Delta t`
    """
    for atom in particles:
        atom.move_linearly(dt)
    
    

def update_velocities(particles, dt, gamma=0): 
    """
    Update the velocities of all particles according to
    
    .. math::
        \\Delta \\vec{v} = \\frac{1}{m} \\vec{F} \\Delta t
        
    If a non-zero gamma is given, a drag force
    :math:`\\vec{F}_\\text{drag} = - \\gamma m \\vec{v}`
    is also applied.

    using :meth:`Particle.accelerate`.
    
    Args:
        particles (list): list of :class:`Particle` objects
        dt (float): time step :math:`\\Delta t`
        gamma (float): coefficient :math:`\\gamma` for the drag force
    """   
    for atom in particles:
        atom.accelerate(dt, gamma)



def langevin_force(particles, dt, gamma, t_external):
    """
    Applies a random Gaussian force to all particles.
    
    In Langevin dynamics, the Newtonian dynamics are extended 
    by adding a drag force
    
    .. math ::
        \\vec{F}_\\text{drag} = - \\gamma m \\vec{v}
        
    and a random Gaussian force :math:`\\vec{F}_\\text{random}` 
    with standard deviation :math:`\\sigma = \\sqrt{ 2 \\gamma m T }`
    and no correlation between forces at different times.
    This function adds such random force to all particles.
    
    Physically, this could represent a system where the simulated particles
    are surrounded by an evironment of other particles that are not explicitly 
    included in the simulation. The drag force represents flow resistance from moving
    through this environment (cf. air resistance). The random force represents
    random collisions between the simulated particles and the 
    particles of the environment (cf. Brownian motion).
    
    This approach also leads to correct sampling of the canonical ensemble 
    at temperature T, so Langevin dynamics can also be used as a thermostat.
    
    Args:
        particles (list): list of :class:`Particle` objects
        dt (float): time step :math:`\\Delta t`
        gamma (float): coefficient :math:`\\gamma` for the drag force
        t_external (float): external temperature
    """

    for atom in particles:
        scaler = sqrt(2.0 * gamma * atom.mass * t_external / dt)
        
        atom.force[0] += scaler * random.standard_normal()
        atom.force[1] += scaler * random.standard_normal()


def velocity_verlet(particles, force_parameters, lattice_parameters, 
        time_parameters, rec_parameters, temperature_parameters):
    """
    Leapfrog version of the Verlet algorithm for integrating the 
    equations of motion, i.e., advancing time.
    
    The algorithm works as follows:
    
    * First, forces are calculated at current time, :math:`\\vec{F}(t)`.
    * Second, velocities are calculated half a time step in the future,
      :math:`\\vec{v}(t + \\frac{1}{2}\\Delta t) = \\vec{v}(t) + \\frac{1}{m}\\vec{F}(t) \\frac{1}{2}\\Delta t`.
    * Then, the following steps are repeated for as long as the simulation runs,
        - Positions are updated by one time step using the velocities,
          :math:`\\vec{r}(t + \\Delta t) = \\vec{r}(t) + \\vec{v}(t + \\frac{1}{2}\\Delta t) \\Delta t`.
        - Forces are calculated using the positions, :math:`\\vec{F}(t + \\Delta t)`    
        - Velocities are updates by one time step using the forces,
          :math:`\\vec{v}(t + \\frac{3}{2}\\Delta t) = \\vec{v}(t + \\frac{1}{2}\\Delta t) + \\frac{1}{m}\\vec{F}(t + \\Delta t) \\Delta t`
    
    Note that the algorithm uses velocities "half a time step from the future" 
    to update positions and forces "half a time step from the future" to update
    the velocities. This approach effectively averages upcoming and previous values
    and leads to a stable algorithm that is symmetric with respect to time reversal.
    
    Args:
        particles (list): list of :class:`Particle` objects
        force_parameters (list): interaction parameters
        lattice_parameters (list): lattice parameters
        time_parameters (list): timing parameters
        rec_parameters (list): recording parameters
        temperature_parameters (list): temperature parameters
        
    Returns:
        float, float: average temperature, pressure
    """
    
    # gather needed parameters
    time = rec_parameters[RECDT_INDEX]
    dt = time_parameters[DT_INDEX]
    trajectory_dt = rec_parameters[ANIDT_INDEX]
    
    t_external = temperature_parameters[T_INDEX]
    gamma = temperature_parameters[TAU_INDEX]
    
            
    # run simulation for this many timesteps    
    steps = int(time/dt)
    
    # record trajectories after this many timesteps have passed
    trajectory_wait = int(trajectory_dt / dt)
    
    calculate_forces(particles, force_parameters, lattice_parameters)
        
    # get velocities at half timestep from beginning for leapfrog
    update_velocities(particles, 0.5*dt, gamma)
    
    t_average = 0.0
    p_average = 0.0
    
    # run the leapfrog algorithm for the required time
    for i in range(steps):
    
        update_positions_without_force(particles, dt)
         
        virial = calculate_forces(particles, force_parameters, lattice_parameters)
        
        t_average += calculate_temperature(particles)
        p_average += calculate_pressure(particles, lattice_parameters, virial)
        
        # apply a thermostat if an external temperature is set
        if t_external > 0:
            langevin_force(particles, dt, gamma, t_external)
        
        update_velocities(particles, dt, gamma)
                
        if i%trajectory_wait == 0:
            for atom in particles:
                atom.wrap(lattice_parameters)
                atom.save_position()            
            
    # velocities are half a timestep in the future, rewind to get correct time
    update_velocities(particles, -0.5*dt, gamma)
    
    
    return t_average/steps, p_average/steps
            
        
        
def randomize_velocities(particles, temperature_parameters):
    """
    Replace the velocities of all particles with random
    velocities drawn from the Maxwell-Boltzmann velocity distribution.
    
    The function makes sure the total momentum from the newly
    assigned velocities is exactly zero.
    
    Args:
        particles (list): list of :class:`Particle` objects
        temperature_parameters (list): temperature parameters
    """

    temperature = temperature_parameters[T_INDEX]
    total_mass = 0.0
    total_momentum = np.array( [0.0, 0.0] )
    
    for atom in particles:
    
        # ignore velocity-constrained particles
        if atom.constraint_type != FREEZE_TAG or atom.constraint_type != V_TAG:
        
            m = atom.mass
            total_mass += m
            
            # for each velocity component, draw a normally distributed
            # random velocity with standard deviation sqrt(T/m)
            for i in range(2):
                v = random.standard_normal() * sqrt(temperature/m)
                atom.velocity[i] = v
                total_momentum[i] += m*v
            
    # make sure the total momentum of the system is zero
    deltav = -total_momentum / total_mass
    for atom in particles:
        atom.velocity += deltav
   
   

def write_system_file(particles, lattice_parameters, filename = "new_system.txt"):
    """
    Write system information in a file.
    
    Args:
        particles (list): list of :class:`Particle` objects
        lattice_parameters (list): lattice parameters
        filename (str): name of the file to write
    """

    file = open(filename, 'w')
        
    file.write("<"+LATTICE_TAG+">\n")
    file.write(X_TAG+": "+str(lattice_parameters[0])+"\n")
    file.write(Y_TAG+": "+str(lattice_parameters[1])+"\n")
    file.write("</"+LATTICE_TAG+">"+"\n"+"\n")
    
    for atom in particles:
        file.write("<"+ATOM_TAG+">\n")

        file.write(INDEX_TAG+": "+str(atom.index)+"\n")
        file.write(NAME_TAG+": "+str(atom.name)+"\n")
        file.write(MASS_TAG+": "+str(atom.mass)+"\n")
        file.write(X_TAG+": "+str(atom.position[0])+"\n")
        file.write(Y_TAG+": "+str(atom.position[1])+"\n")
        file.write(VX_TAG+": "+str(atom.velocity[0])+"\n")
        file.write(VY_TAG+": "+str(atom.velocity[1])+"\n")

        for atom_B in atom.connected_atoms:
            file.write(CONNECT_TAG+": "+str(atom_B.index)+"\n")   

        file.write("</"+ATOM_TAG+">\n"+"\n")
        
    file.close()
   
   
def print_progress(step, total):
    """
    Prints a progress bar.
    
    Args:
        step (int): progress counter
        total (int): counter at completion
    """

    message = "simulation progress ["
    total_bar_length = 60
    percentage = int(step / total * 100)
    bar_fill = int(step / total * total_bar_length)
    for i in range(total_bar_length):
        if i < bar_fill:
            message += "|"
        else:
            message += " "
    
    message += "] "+str(percentage)+" %"
    if step < total:
        print(message, end="\r")     
    else:
        print(message) 
    


def run_simulation(particles, 
                   force_parameters, 
                   lattice_parameters, 
                   time_parameters,
                   rec_parameters,
                   temperature_parameters ):
    """
    Run a molecular dynamics simulation.
    
    Args:
        particles (list): list of :class:`Particle` objects
        force_parameters (list): interaction parameters
        lattice_parameters (list): lattice parameters
        time_parameters (list): timing parameters
        rec_parameters (list): recording parameters
        temperature_parameters (list): temperature parameters
    
    Returns: 
        tuple: arrays containing measured physical quantities at different times,
             (time, temperature, pressure, momentum, kinetic energy, spring energy, LJ energy)
    """


    # lists for recording statistics - record starting values
    times = [ ]
    temperatures = [ ]
    pressures = [ ]
    momenta = [ ]
    kinetic_energies = [ ]
    spring_energies = [ ]
    lj_energies = [ ]
    
    
    # gather needed parameters
    runtime = time_parameters[RUNTIME_INDEX] # total simulation time
    sample_interval = rec_parameters[RECDT_INDEX] # simulation time in each sample
    timestep = time_parameters[DT_INDEX] # timestep used for simulation

    # simulation will be split in n_samples pieces for statistics    
    n_samples = int(runtime/sample_interval)
    true_sample_time = int(sample_interval / timestep) * timestep
    
    # run the simulation in n_samples pieces
    for i in range(n_samples):
    
        #print("running sample "+str(i+1)+" / "+str(n_samples))
        print_progress(i, n_samples)
    
        # run the simulation for the required length
        temp, pres = velocity_verlet(particles, force_parameters, lattice_parameters, 
                        time_parameters, rec_parameters, temperature_parameters)
    
        # record physical quantities
        times.append( (i+0.5) * true_sample_time ) 
        momenta.append( calculate_momentum(particles) ) 
        temperatures.append( temp )
        pressures.append( pres )
        kinetic_energies.append( calculate_kinetic_energy(particles) )
        spring_energies.append( calculate_spring_potential_energy(particles, force_parameters, lattice_parameters) )
        lj_energies.append( calculate_lj_potential_energy(particles, force_parameters, lattice_parameters) )


    print_progress(n_samples, n_samples)

    times = np.array(times)
    momenta = np.array(momenta)
    temperatures = np.array(temperatures)
    pressures = np.array(pressures)
    kinetic_energies = np.array(kinetic_energies)
    spring_energies = np.array(spring_energies)
    lj_energies = np.array(lj_energies)

    return times, temperatures, pressures, momenta, kinetic_energies, spring_energies, lj_energies




def calculate_avr_std_err(values, start_index=0, n_split=5):
    """
    Calculates the average and standard error of mean of a sequence.
    
    The values may be correlated so the algorithm divides the
    sequence in shorter subsequences and uses the averages of these sequences
    to calculate the error.
    
    If the beginning of the sequence cannot be used in the analysis (equilibrium
    has not yet been reached), one can ignore the early values by specifying a
    starting index.
    
    Args:
        values (array): values to analyse
        start_index (int): index of the first value to be included in the analysis
        n_split (int): number of subsequences used for error calculation
    """
    avr = 0.0
    var = 0.0
    
    # start looking at values after start_index
    sequence = values[start_index:]    
    len_total = len(sequence)   

    len_split = int( len_total / n_split )
    avr_split = [0] * n_split    
    
    # split the sequence in n_split parts
    for i in range(n_split):
    
        # the part we are currently looking at
        short_sequence = sequence[i*len_split : (i+1)*len_split]
        for value in short_sequence:

            # calculate the average for this part
            avr_split[i] += value/len_split
    
        # calculate the total average
        avr += avr_split[i] / n_split    
    
    # calculate sample variance using the averages of the parts
    for a in avr_split:
    
        var += (a-avr)**2/(n_split-1)
      
    # standard deviation
    std = sqrt( var )
    
    # standard error of mean
    error = std / sqrt(n_split)
        
    return avr, std, error
    
    
def calculate_average_and_error(values, start=0):
    """
    Calculates the average and standard error of mean of a sequence.
    
    The values in the sequence are assumed to be uncorrelated.
    
    If the beginning of the sequence cannot be used in the analysis (equilibrium
    has not yet been reached), one can ignore the early values by specifying a
    starting index.
    
    Args:
        values (array): values to analyse
        start (int): index of the first value to be included in the analysis
    """

    avr_x = 0.0
    avr_sq = 0.0
    for x in values[start:]:
        avr_x += x
        avr_sq += x*x
        
    n = float(len(values)-start)
    if n > 0:
        avr_x /= n
        avr_sq /= n
        variance = (avr_sq - avr_x*avr_x)*n/(n-1)
        error = sqrt(variance/n)
    else:
        error = 0.0
    
    return avr_x, error
    


def main(system_file = "system.txt", simu_file = "simulation.txt"):
    """
    The main program.
    
    Reads system and simulation information from files,
    runs a simulation, and calculates statistics.
    
    Possibly also shows an animation of the simulation.
    
    Args:
        system_file (str): name of the file containing system info
        simu_file (str): name of the file containing physical and simulation info
    """
    
    lattice_parameters, particles = read_system(system_file)
    force_parameters, temperature_parameters = read_physics(simu_file)    
    time_parameters, rec_parameters = read_timescale(simu_file)
    constraints = read_constraints(simu_file)
    apply_constraints(particles, constraints)   
    
    # randomize velocities?
    random_start = temperature_parameters[RS_INDEX]
    if random_start == "yes" or random_start == "y":
        randomize_velocities(particles, temperature_parameters)    
    
    start_time = time.perf_counter()
    # run the simulation
    ts, temps, Ps, ps, ks, uss, uljs = run_simulation(  particles, 
                                                    force_parameters, 
                                                    lattice_parameters, 
                                                    time_parameters, 
                                                    rec_parameters,
                                                    temperature_parameters )
    end_time = time.perf_counter()
    
    write_system_file( particles, lattice_parameters )
    print("simulation time: "+str(end_time-start_time)+" s")

        
    # calculate all averages and errors
    #
    # ignore the start of the run as a thermalization period
    thermal_time = rec_parameters[THERMAL_INDEX]
    total_time = time_parameters[RUNTIME_INDEX]    
    sampling_start = int( len(ts) * thermal_time / total_time )
    
    vavr = lattice_parameters[0]*lattice_parameters[1]
    tavr, terr = calculate_average_and_error(temps, sampling_start)
    Pavr, Perr = calculate_average_and_error(Ps, sampling_start)
    pxavr, pxerr = calculate_average_and_error(ps[:,0], sampling_start)
    pyavr, pyerr = calculate_average_and_error(ps[:,1], sampling_start)
    kavr, kerr = calculate_average_and_error(ks, sampling_start)
    usavr, userr = calculate_average_and_error(uss, sampling_start)
    uravr, urerr = calculate_average_and_error(uljs, sampling_start)
    
    # print measured values, use 2 * error of mean as the confidence interval
    acc = 3 # decimals to print
    print("volume         = "+str(round(vavr,acc)) )
    print("atoms          = "+str(len(particles)) )
    print("temperature    = "+str(round(tavr,acc))+" +- "+str(round(2*terr,acc)) )
    print("pressure       = "+str(round(Pavr,2*acc))+" +- "+str(round(2*Perr,2*acc)) )
    print("momentum(x)    = "+str(round(pxavr,acc))+" +- "+str(round(2*pxerr,acc)) )
    print("momentum(y)    = "+str(round(pyavr,acc))+" +- "+str(round(2*pyerr,acc)) )
    print("kinetic energy = "+str(round(kavr,acc))+" +- "+str(round(2*kerr,acc)) )
    print("spring energy  = "+str(round(usavr,acc))+" +- "+str(round(2*userr,acc)) )
    print("LJ energy      = "+str(round(uravr,acc))+" +- "+str(round(2*urerr,acc)) )

    # animate?
    ap = rec_parameters[PLANE_INDEX]
    if ap == 'yes' or ap == 'y':
        animate(particles, lattice_parameters, force_parameters)

    # plot?
    pl = rec_parameters[PLOT_INDEX]
    if pl == "yes" or pl == "y":

        plot_function_and_average(ts, temps, tavr, 2*terr, "temperature")
        plot_function_and_average(ts, Ps, Pavr, 2*Perr, "pressure")

        # plot energies
        plt.plot(ts, ks, label="kinetic energy")
        plt.plot(ts, uss + uljs, label="potential energy" )
        plt.plot(ts, ks + uss + uljs, label="total energy" ) 
        plt.legend() 
        plt.xlabel("t")  
        plt.ylabel("E") 
        plt.show()

    



def plot_function_and_average(time, values, average, error, ylabel, show=True):
    """
    Plots a time series of a quantity as well as its average and error marginal.
    
    Args:
        time (array): time at each data point
        values (array): recorded value at each data point
        average (float): average of the recorded values
        error (float): error estimate for the average
        ylabel (str): name of the variable
        show (bool): Of True, the plot is shown on screen, otherwise only created in memory.
    """
    plt.plot(time, average*np.ones(len(time)), 'b:')
    plt.fill_between(time, (average-error)*np.ones(len(time)), (average+error)*np.ones(len(time)), color = 'b', alpha=0.1 )
    plt.plot(time, values, 'r')
    plt.xlabel("time")
    plt.ylabel(ylabel)
    if show:
        plt.show()
    

    
if __name__ == "__main__":
    random = default_rng()
    
    main("A4_gas_system.txt", "A4_gas_simulation.txt")
    #main("A4_bridge_system.txt", "A4_bridge_simulation.txt")
    #main("2d_bridge_E2.txt", "A4_bridge_simulation.txt")
    #main("2d_bridge_test.txt", "A4_bridge_simulation.txt")
  
else:
    random = default_rng()      
    