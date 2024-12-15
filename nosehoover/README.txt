nosehoover.py by Otso Santala

This program implements the Nosé-Hoover thermostat to an 
existing MD program made by Teemu Hynninen (A4_2dmd_working.py). The general formulation is
based on the one given by K. Vollmayer-Lee  (Am. J. Phys 88, 401-422 (2020), https://doi.org/10.1119/10.0000654), 
originally by W. G. Hoover, (Phys. Rev. A 31, 1695–1697 (1985)).

Included in this is the A4_2dmd_working.py, which is needed for the program to function, as well as a simulation parameter file,
'simulation_file.txt', and two system files, 'system_file_25.txt' and 'system_file_49.txt', which define a 50 x 50 system with 25
or 49 atoms arranged in a square lattice, respectively. The parameters set in the simulation file are the ones used in the final 
report, but feel free to vary them. Thermostat is chosen with a separate tag in the simulation file, and a minor tweaking parameter
can also be used. 

To change the desired system, you must go to the main program call right at the bottom of the source code and comment/uncomment the
required lines. You can also set the desired input and output files here, if you like. By default, the inputs are 'system_file_25.txt'
for the system and 'simulation_file.txt' for the simulation parameters, and the output is 'simulation_data.txt', which will appear
in the same directory where the program is located.