import numpy as np
import A4_2dmd_working as md
import time


TS_TAG = "thermostat"
ADJUST_TAG = "initial-adjust"

TS_INDEX = 0
ADJUST_INDEX = 1



def read_thermostat_and_adjust(filename):
    """
    Reads the simulation file and returns values for type of thermostatting and 
    whether initial positions are to be adjusted.

    Arguments:
        filename (str): simulation file

    Returns:
        params (list): list with thermostat and adjust parameters
    """
    file = open(filename, "r")
    lines = file.readlines()

    infos, dummy = md.find_info(lines, md.TEMP_TAG)

    params = [0]*2
    for line in infos:
        tag, inf = md.parse_line(line)
        if tag == TS_TAG:
            params[TS_INDEX] = str(inf)
        elif tag == ADJUST_TAG:
            params[ADJUST_INDEX] = float(inf)

    return params

        
def berendsen_thermostat(particles, dt, tau, t_external):
    """
    Performs Berendsen thermostatting. Velocities are scaled according to a scaling factor

        :math:`\\lambda = \\sqrt{1 + \\frac{\\Delta t}{\\tau}(\\frac{T_0}{T} - 1)}`,

    where :math:`\\Delta t` is the time step, :math:`\\tau` is a time coupling constant, :math:`T_0` is the temperature
    of the external heat bath and :math:`T` is the current temperature of the system.

    Arguments:
        particles (list): a list of :class:`Particle` type objects
        dt (float): time step
        tau (float): coupling time constant
        t_external (float): external temperature
    """

    t = md.calculate_temperature(particles)
    lamb = np.sqrt(1 + dt / tau * (t_external / t - 1))
    for atom in particles:
        atom.velocity *= lamb
    

def calculate_xi_dot(particles, Q, T):
    """
    Calculates the first time derivative of the friction coefficient.

    Arguments:
        particles (list): list of objects of type :class:`Particle`
        Q (float): thermal inertia (coupling parameter)
        T (float): temperature of the heat bath
    Returns:
        xi_dot (float): time derivative of the friction coefficient
    """
    k_total = md.calculate_kinetic_energy(particles) 
    xi_dot = 1/Q*(2*k_total - 2 * len(particles) * T)
    return xi_dot



def update_positions_fa(particles, xi, dt):
    """
    Updates the positions of particles in the Fox-Andersen integration scheme.

    Arguments:
        particles (list): list of objects of type :class:`Particle`
        xi (float): friction coefficient at :math:`t`
        dt (float): time step
    """
    for atom in particles:
        atom.position += dt * atom.velocity + 0.5 * dt**2 * (atom.force / atom.mass - xi * atom.velocity)

    

def update_velocities_fa(particles, prev_forces, xi, next_xi, dt):
    """
    Updates the velocities of particles in the Fox-Andersen integration scheme.

    Arguments:
        particles (list): list of objects of type :class:`Particle`
        prev_forces (dict): a dictionary for forces on each atom in the previous timestep
        xi (float): friction coefficient at :math:`t`
        next_xi (float): friction coeffcient at :math:`t + \\Delta t`
        dt (float): time step
    """
    for atom in particles:
        atom.velocity += dt / 2 * ((prev_forces[atom] + atom.force) / atom.mass - (xi + next_xi) * atom.velocity) * (1 - dt / 2 * next_xi)



def update_xi_fa(particles, Q, T, xi, dt):
    """
    Calculates the friction coefficient in the Fox-Andersen integration scheme.

    Arguments:
        particles (list): list of objects of type :class:`Particle`
        Q (float): thermal inertia (coupling parameter)
        T (float): temperature of the heat bath
        xi (float): friction coefficient at :math:`t`
        dt (float): time step
    Returns:
        next_xi (float): friction coeffcient at :math:`t + \\Delta t`
    """
    xi_dot = calculate_xi_dot(particles, Q, T)
    next_xi = xi + dt * xi_dot
    return next_xi



def fox_andersen(particles, force_params, lattice_params, time_params, rec_params, temp_params):
    """
    (based on velocity_verlet() in A4_2dmd_working.py by Teemu Hynninen)

    Performs Fox-Andersen integration of the Nosé-Hoover equations on the target system. Nosé-Hoover
    equations are as follows:

        :math:`\\ddot{\\vec{r}} = \\frac{1}{m}\\vec{F} - \\xi\\dot{\\vec{r}}`

        :math:`\\dot{\\xi} = \\frac{1}{Q} \\left ( \\sum_{i = 1}^{N} m_i \\dot{r}_i^2 - dN k_b T_0 \\right )`.

    Steps in Fox-Andersen integration:

        Position at :math:`t + \\Delta t`:
            :math:`\\vec{r}(t + \\Delta t) = \\vec{r}}(t) + \\dot{\\vec{r}}(t) \\Delta t + \\frac{1}{2}\\left [\\frac{\\vec{F}}{m} - \\xi\\vec{r}}(t) \\right ] (\\Delta t)^2`.

        Velocity at :math:`t + \\Delta t`:
            :math:`\\dot{\\vec{r}}(t + \\Delta t) = \\dot{\\vec{r}}(t) + \\frac{\\Delta t}{2} \\left {  \\frac{\\vec{F}(t)`
            :math:`+ \\vec{F}(t+\\Delta t)}{m} - [\\xi(t) + \\xi(t+\\Delta t)]\\dot{\\vec{r}}(t)  \\right}\\left[1 - \\frac{\\Delta t}{2}\\xi(t + \\Delta t) \\right]`.

    The friction coefficient at :math:`t + \\Delta t` is approximated as :math:`\\xi(t + \\Delta t) \\approx \\xi(t) + \\dot{\\xi}(t)\\Delta t`.


    Arguments:
        particles (list): list of :class:`Particle` objects
        force_params (list): interaction parameters
        lattice_params (list): lattice parameters
        time_params (list): timing parameters
        rec_params (list): recording parameters
        t_params (list): temperature parameters
        
    Returns:
        float, float: average temperature, pressure
    """

    # Timescale and recording parameters
    # The simulation is run in parts: function returns averages calculated
    # over sample_dt intervals
    dt = time_params[md.DT_INDEX]
    sample_dt = rec_params[md.RECDT_INDEX]
    animation_dt = rec_params[md.ANIDT_INDEX] # trajectory recording
    steps = int(sample_dt / dt)
    
    # Temperature parameters
    t_ext = temp_params[md.T_INDEX]
    q = temp_params[md.TAU_INDEX]

    # initialize the quantities
    xi = 0.0
    t_avg = 0.0
    p_avg = 0.0

    # calculate initial forces
    md.calculate_forces(particles, force_params, lattice_params)

    # keep track of previous forces using a dictionary
    prev_forces = {}
    for atom in particles:
        prev_forces[atom] = atom.force
    
    # no need for half steps in this integrator, so go straight to looping
    for i in range(steps):
        update_positions_fa(particles, xi, dt)

        # Calculate approximate xi at the next timestep
        next_xi = update_xi_fa(particles, q, t_ext, xi, dt)

        # update forces and get the virial for pressure calculations
        virial = md.calculate_forces(particles, force_params, lattice_params)

        update_velocities_fa(particles, prev_forces, xi, next_xi, dt)

        # update xi
        xi = next_xi

        # update previous forces and record trajectories occasionally
        for atom in particles:
            prev_forces[atom] = atom.force

            # this is the same as in velocity_verlet(). If step == animation step interval, then
            # wrap atoms to the main lattice and save their positions
            if i % int(animation_dt / dt) == 0:
                atom.wrap(lattice_params)
                atom.save_position()

        # Calculate temperatures and pressures
        t_avg += md.calculate_temperature(particles)
        p_avg += md.calculate_pressure(particles, lattice_params, virial)

        # This is for checking if the simulation blows up (quantities are NaN), 
        # which saves from the hassle of interrupting with Ctrl+C.
        if np.isnan(t_avg):
            raise Exception("\nSimulation blew up!\nProgram finished.")
        
    return t_avg / steps, p_avg / steps




def leapfrog_verlet(particles, force_parameters, lattice_parameters, 
        time_parameters, rec_parameters, temperature_parameters, thermostat):
    """
    (modified from velocity_verlet() in A4_2dmd_working.py, originally by Teemu Hynninen)

    Leapfrog version of the Verlet algorithm for integrating the 
    equations of motion, i.e., advancing time, either with Langevin dynamics (ld) or Berendsen thermostat (be).
    
    Arguments:
        particles (list): list of :class:`Particle` objects
        force_parameters (list): interaction parameters
        lattice_parameters (list): lattice parameters
        time_parameters (list): timing parameters
        rec_parameters (list): recording parameters
        temperature_parameters (list): temperature parameters
        thermostat (string): type of thermostatting
        
    Returns:
        float, float: average temperature, pressure
    """
    
    # gather needed parameters
    time = rec_parameters[md.RECDT_INDEX]
    dt = time_parameters[md.DT_INDEX]
    trajectory_dt = rec_parameters[md.ANIDT_INDEX]

    t_external = temperature_parameters[md.T_INDEX]

    # tau in Berendsen, gamma in Langevin
    t_param = temperature_parameters[md.TAU_INDEX]
    
    # run simulation for this many timesteps    
    steps = int(time/dt)
    
    # record trajectories after this many timesteps have passed
    trajectory_wait = int(trajectory_dt / dt)
    
    md.calculate_forces(particles, force_parameters, lattice_parameters)

    # get velocities at half timestep from beginning for leapfrog
    if thermostat == "be" or thermostat == "berendsen":
        # leave out the temperature parameter for Berendsen
        md.update_velocities(particles, 0.5*dt)

    elif thermostat == "ld" or thermostat == "langevin":
        # if gamma > 0, then Langevin dynamics
        md.update_velocities(particles, 0.5*dt, t_param)

    t_average = 0.0
    p_average = 0.0
    
    # run the leapfrog algorithm for the required time
    for i in range(steps):
    
        md.update_positions_without_force(particles, dt)
         
        virial = md.calculate_forces(particles, force_parameters, lattice_parameters)
        
        t_average += md.calculate_temperature(particles)
        p_average += md.calculate_pressure(particles, lattice_parameters, virial)
        
        if np.isnan(t_average):
            raise ValueError("\nSimulation blew up!\nProgram finished.")
            
        # apply a thermostat if an external temperature is set
        if thermostat == "be" or thermostat == "berendsen":
            if t_external > 0:
                berendsen_thermostat(particles, dt, t_param, t_external)
            md.update_velocities(particles, dt)

        elif thermostat == "ld" or thermostat == "langevin":
            if t_external > 0:
                md.langevin_force(particles, dt, t_param, t_external)
            md.update_velocities(particles, dt, t_param)

        if i%trajectory_wait == 0:
            for atom in particles:
                atom.wrap(lattice_parameters)
                atom.save_position()            
            
    # velocities are half a timestep in the future, rewind to get correct time
    if thermostat == "be":
        md.update_velocities(particles, -0.5*dt)

    elif thermostat == "ld":
        md.update_velocities(particles, -0.5*dt, t_param)
    
    
    return t_average/steps, p_average/steps

def nonidealize(particles, max_adj):
    """
    Makes minor adjustments to the positions of the particles.

    Arguments:
        particles (list): list of objects of type :class:`Particle`
        max_adj (float): maximum amount of adjustment applied per coordinate
    """
    random = np.random.default_rng()
    for atom in particles:
        atom.position += random.uniform(-max_adj, max_adj, 2)


def write_data(particles, lattice_params, force_params, temp_params,
                rec_params, time_params, thermostat, quantities, avg_quantities, errors, sys_file, output_file):
    """
    Appends simulation data to an output file.

    Arguments:
        particles (list): list of :class:`Particle` objects
        lattice_params (list): lattice parameters
        force_params (list): interaction parameters
        temp_params (list): temperature parameters
        rec_params (list): recording parameters
        time_params (list): timescale parameters
        thermostat (str): thermostatting used
        quantities (list): list of quantities to record
        avg_quantities (list): list of averages of quantities
        errors (list): list of standard error of means
        sys_file (str): system file
        output_file (str): output file
    """
    file = open(output_file, "a")

    # These are now declared for clarity
    if thermostat == "nh":
        thermostat = "Nose-Hoover"
        coupling_param = "Q"

    elif thermostat == "ld":
        thermostat = "Langevin dynamics"
        coupling_param = "gamma"

    elif thermostat == "be":
        thermostat = "Berendsen"
        coupling_param = "tau"

    # Probably a better way to do this, but gets the job done
    file.write(f"########## Simulation of '{sys_file}' on " + time.strftime("%d.%m.%Y %H:%M:%S")+  " ##########\n\n")
    file.write(f"Thermostat: {thermostat} with {coupling_param} = {temp_params[md.TAU_INDEX]}\n")
    file.write(f"External temperature: {temp_params[md.T_INDEX]}\n")
    file.write(f"Random start: {temp_params[md.RS_INDEX]}\n")
    file.write(f"Atoms: {len(particles)}\n")
    file.write(f"System size: {lattice_params[0]} x {lattice_params[1]}\n")
    file.write(f"LJ parameters: sigma = {force_params[md.SIGMA_INDEX]}, epsilon = {force_params[md.EPS_INDEX]}\n")
    file.write(f"Time step: {time_params[md.DT_INDEX]}\n")
    file.write(f"Total runtime: {time_params[md.RUNTIME_INDEX]}\n")
    file.write(f"Thermalizing time: {rec_params[md.THERMAL_INDEX]}\n\n")
    file.write("Time\tTemperature\tPressure\tMomentum\tKinetic energy\tPot. energy\tTotal energy\n")

    # Write the proper data
    for j in range(len(quantities[0])):
        for i in range(len(quantities)):
            file.write(f"{quantities[i][j]:<7.3f}\t")
        file.write("\n")
    
    # Write averages and errors
    file.write("Averages:\nTemperature\tPressure\tMomentum\tKinetic energy\tPot. energy\tTotal energy\n")
    for i in range(len(avg_quantities)):
        file.write(f"{avg_quantities[i]:<.3f} +- {errors[i]:<.3f}\t")
    file.write("\n\n\n")
    file.close()
        

def run_simulation(particles, 
                   force_parameters, 
                   lattice_parameters, 
                   time_parameters,
                   rec_parameters,
                   temperature_parameters, thermostat="nh"):
    """
    (Modified from run_simulation() in A4_2dmd_working.py, originally by Teemu Hynninen)

    Run a molecular dynamics simulation. 
    
    Arguments:
        particles (list): list of :class:`Particle` objects
        force_parameters (list): interaction parameters
        lattice_parameters (list): lattice parameters
        time_parameters (list): timing parameters
        rec_parameters (list): recording parameters
        temperature_parameters (list): temperature parameters
        thermostat (string): type of thermostatting
    
    Returns: 
        tuple: arrays containing measured physical quantities at different times,
             (time, temperature, pressure, momentum, kinetic energy, LJ energy)
    """

    print(f"Setting up simulation with thermostat type '{thermostat}'")

    # lists for recording statistics - record starting values
    times = [ ]
    temperatures = [ ]
    pressures = [ ]
    momenta = [ ]
    kinetic_energies = [ ]
    lj_energies = [ ]
    

    # gather needed parameters
    runtime = time_parameters[md.RUNTIME_INDEX] # total simulation time
    sample_interval = rec_parameters[md.RECDT_INDEX] # simulation time in each sample
    timestep = time_parameters[md.DT_INDEX] # timestep used for simulation

    # simulation will be split in n_samples pieces for statistics    
    n_samples = int(runtime/sample_interval)
    true_sample_time = int(sample_interval / timestep) * timestep
    

    # run the simulation in n_samples pieces
    for i in range(n_samples):
    
        #print("running sample "+str(i+1)+" / "+str(n_samples))
        md.print_progress(i, n_samples)

        # Nosé-Hoover thermostatting uses Fox-Andersen integration
        if thermostat == "nh":
            temp, pres = fox_andersen(particles, 
                                      force_parameters, 
                                      lattice_parameters, 
                                      time_parameters, 
                                      rec_parameters, 
                                      temperature_parameters)

        # Langevin or Berendsen use Verlet
        elif thermostat == "ld" or thermostat == "be":
            temp, pres = leapfrog_verlet(particles, 
                                        force_parameters, 
                                        lattice_parameters, 
                                        time_parameters, 
                                        rec_parameters, 
                                        temperature_parameters, thermostat)
            

        # record physical quantities
        times.append( (i+0.5) * true_sample_time ) 
        momenta.append( md.calculate_momentum(particles) ) 
        temperatures.append( temp )
        pressures.append( pres )
        kinetic_energies.append( md.calculate_kinetic_energy(particles) )
        lj_energies.append( md.calculate_lj_potential_energy(particles, force_parameters, lattice_parameters) )


    md.print_progress(n_samples, n_samples)

    times = np.array(times)
    momenta = np.array(momenta)
    temperatures = np.array(temperatures)
    pressures = np.array(pressures)
    kinetic_energies = np.array(kinetic_energies)
    lj_energies = np.array(lj_energies)

    return times, temperatures, pressures, momenta, kinetic_energies, lj_energies


def main(system_file = "system_file.txt", simulation_file = "simulation_file.txt", output_file = "simulation_data.txt"):
    """
    The main program, based on main() in A4_2dmd_working.py by Teemu Hynninen.

    Simulates the system described in input files, optionally plots time averages of temperatures, pressures,
    momenta, total energies. Optionally animates the trajectories. Writes the simulation data to an output file.

    Arguments:
        system_file (str): system size definitions
        simulation_file (str): system parameters
        output_file (str): file where data is written
    """
    # Read parameters from the system and simulation files.

    lattice_params, particles = md.read_system(system_file)
    force_params, temp_params = md.read_physics(simulation_file)
    time_params, rec_params = md.read_timescale(simulation_file)
    therms_adj_params = read_thermostat_and_adjust(simulation_file)   # thermostat and adjustment parameters
    
    # The given system files have atoms in a square grid. If this is too ideal,
    # this adjusts initial positions, when a positive adjustment parameter is given.

    if therms_adj_params[ADJUST_INDEX] > 0:
        nonidealize(particles, therms_adj_params[ADJUST_INDEX])

    # FOR CONTROLLED TESTING: write a new system file from the adjusted configuration
    #md.write_system_file(particles, lattice_params, "adjusted_system.txt")
    #return 0

    # Randomize velocities, if random start is set in simulation file.
    if temp_params[md.RS_INDEX] == "yes":
        md.randomize_velocities(particles, temp_params)

    # Run the simulation with the chosen thermostat
    thermostat = therms_adj_params[TS_INDEX]

    times, temps, press, momenta, k_ergs, p_ergs = run_simulation(particles,
                                                                    force_params, 
                                                                    lattice_params, 
                                                                    time_params, 
                                                                    rec_params, 
                                                                    temp_params, 
                                                                    thermostat)
    
    
    # Calculate average quantities and standard errors of means after thermalization
    # and plot them

    thermal_time = rec_params[md.THERMAL_INDEX] 
    total_time = time_params[md.RUNTIME_INDEX]    
    start_index = int( len(times) * thermal_time / total_time )

    temp_avg, temp_err = md.calculate_average_and_error(temps, start_index)
    press_avg, press_err = md.calculate_average_and_error(press, start_index)

    # instead of x and y components, just use the total momentum of the system. This should differ from zero
    # only with Langevin dynamics

    total_momenta = []
    for momentum in momenta:
        total_momentum = np.sqrt(momentum @ momentum)
        total_momenta.append(total_momentum)
    total_momenta = np.array(total_momenta)

    momentum_avg, momentum_err = md.calculate_average_and_error(total_momenta, start_index)
    k_avg, k_err = md.calculate_average_and_error(k_ergs, start_index)
    p_avg, p_err = md.calculate_average_and_error(p_ergs, start_index)

    # potential energies are so small compared to kinetic energies, so plot only the total energies
    total_ergs = k_ergs + p_ergs
    e_avg, e_err = md.calculate_average_and_error(total_ergs, start_index)

    # plot functions and averages with 95% confidence interval
    if rec_params[md.PLOT_INDEX] == "yes":
        md.plot_function_and_average(times, temps, temp_avg, 1.96*temp_err, "Temperature, avg", True)
        md.plot_function_and_average(times, press, press_avg, 1.96*press_err, "Pressure, avg", True)
        md.plot_function_and_average(times, total_momenta, momentum_avg, 1.96*momentum_err, "Momentum, avg", True)
        md.plot_function_and_average(times, total_ergs, e_avg, 1.96*e_err, "Total energy, avg", True)

    # animation
    if rec_params[md.PLANE_INDEX] == "yes":
        md.animate(particles, lattice_params, force_params)

    # write data to a file for later analysis
    quantities = [times, temps, press, total_momenta, k_ergs, p_ergs, total_ergs]
    avg_quantities = [temp_avg, press_avg, momentum_avg, k_avg, p_avg, e_avg]
    errors = [temp_err, press_err, momentum_err, k_err, p_err, e_err]

    write_data(particles, lattice_params, force_params, temp_params,
                rec_params, time_params, thermostat, quantities, avg_quantities, errors, system_file, output_file)
    
    
        
if __name__ == "__main__":
    # Two system files are provided, adjust parameters in simulation_file.txt

    main(system_file="system_file_25.txt")
    main(system_file="system_file_49.txt")

    print("Program finished.")
    
       
    