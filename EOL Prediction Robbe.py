##### End-of-Life Prediction #####
# Load standard modules
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import winsound

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime, datetime_to_tudat, date_time_from_epoch, datetime_to_python

# Load modules for getting TLE
import requests
import re
from datetime import datetime, timedelta
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

def get_tle(norad_cat_id, date):
    # INPUTS: norad_cat_id (int) = ID of the satellite, date (str) = Date to get TLE at in form YEAR-MONTH-DAY--YEAR-MONTH-DAY
    # EXAMPLE: get_tle(32789, 2022-09-06)
    # OUTPUT: TLE in lines one and two for entering into tudatpy TLE

    # Define the login data
    login_data = {
        'identity': 'wschaerlaecken@gmail.com',
        'password': 'groupd03123456789'}
    
    NORAD_CAT_ID = norad_cat_id
    DATE = date
    # Create a session
    with requests.Session() as session:
        # Post the login data
        post_response = session.post('https://www.space-track.org/ajaxauth/login', data=login_data)

        # Check if login was successful
        if post_response.status_code == 200:
            # If login is successful, make the GET request
            url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{NORAD_CAT_ID}/orderby/TLE_LINE1%20ASC/EPOCH/{DATE}/format/tle"
            #print(url)
            get_response = session.get(url)

            if get_response.status_code == 200:
                data = get_response.text
                #print(data)
            else:
                print("Failed to retrieve data. Status code:", get_response.status_code)
                print("Response text:", get_response.text)
        else:
            print("Failed to log in. Status code:", post_response.status_code)
            print("Response text:", post_response.text)

        # Split the data into individual lines
        lines = data.split('\n')
        line1 = lines[0]
        line2 = lines[1]

        # Iterate over each line and assign the values to variables
        for line in lines:
            if line.startswith('1 '):
                line1split = line.split(' ')
            elif line.startswith('2 '):
                line2split = line.split(' ')

        for line in lines[:2]:
            if line.startswith('1 '):
                line1split = re.findall(r'\d+\.\d+|\d+', line)  # Find all numeric characters and decimal points
            elif line.startswith('2 '):
                line2split = re.findall(r'\d+\.\d+|\d+', line)  # Find all numeric characters and decimal points

        return line1, line2, line1split, line2split

# Load spice kernels
spice.load_standard_kernels()


# Useful datasets: [name, norad_cat_id, mass, reference area, drag coefficient, radiation pressure coefficient, launch date]
C3_data =   ["Delfi-C3",   32789, 2.2, 0.080, 1.7, 1.1,  "2008-04-28"]
N3XT_data = ["Delfi-N3XT", 39428, 2.8, 0.087, 2.2, 1.3,  "2013-11-21"]
PQ_data =   ["Delfi-PQ",   51074, 0.6, 0.011, 2.2, 1.3,  "2022-01-13"]

###########################################################################################
##### SETUP VARIABLES #####################################################################
dataset = C3_data                                   # For automatic data input
tle_date = "2022-09-06--2022-09-07"                 # Date for TLE
propagation_duration = 6000                         # How long to propagate for at most [days]
fixed_step_size = 100.0                             # Step size for integrator

# Set manually if needed, otherwise change dataset
satellite = dataset[0]                              # Satellite name
satellite_norad_cat_id = dataset[1]                 # NORAD catelog ID for TLE
satellite_mass = dataset[2]                         # Mass of satellite [kg]
reference_area = dataset[3]                         # Reference area for aerodynamics [m²]
drag_coefficient = dataset[4]                       # Drag coefficient [-]
reference_area_radiation = dataset[3]               # Reference area for radiation pressure [m²]
radiation_pressure_coefficient = dataset[5]         # Radiation pressure coefficient [-]
#####^ SETUP VARIABLES ^###################################################################
###########################################################################################

# Get TLE in two lines
line1, line2, line1split, line2split = get_tle(satellite_norad_cat_id, tle_date)
line1 = line1.replace("\r","")
line2 = line2.replace("\r","")
#print(line1, "\n", line2, "\n", line1split, "\n", line2split)

# Get starting date and end dates for propagation
Epoch_Year = line1split[3][:2]
Epoch_Day = line1split[3][2:]
if int(Epoch_Year) < 30:
    epoch_year = int(Epoch_Year) + 2000  # Add 2000 to get the full year
else:
    epoch_year = int(Epoch_Year) + 1900  
start_of_year = datetime(epoch_year, 1, 1)  # January 1 of the epoch year
epoch_day = float(Epoch_Day)  # Convert to float to handle fractional days
date1 = start_of_year + timedelta(days=epoch_day - 1) # Subtract 1 because the epoch day is 1-indexed, but timedelta days are 0-indexed
date2 = date1 + timedelta(days=propagation_duration)

print(f"Propagation starting date: {date1}")

# Set simulation start and end epochs
simulation_start_epoch = datetime_to_tudat(date1).epoch()
simulation_end_epoch   = datetime_to_tudat(date2).epoch()

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()
atmomodel = "NRLMSISE-00"

# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle objects.
bodies.create_empty_body(satellite)
bodies.get(satellite).mass = satellite_mass

# Create aerodynamic coefficient interface settings, and add to vehicle
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area, [drag_coefficient, 0, 0]
)
environment_setup.add_aerodynamic_coefficient_interface(
    bodies, satellite, aero_coefficient_settings)

# Create radiation pressure settings, and add to vehicle
occulting_bodies_dict = dict()
occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

environment_setup.add_radiation_pressure_target_model(
    bodies, satellite, vehicle_target_settings)

# Define bodies that are propagated
bodies_to_propagate = [satellite]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on the sateliite by Sun and Earth.
accelerations_settings_satellite = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
        propagation_setup.acceleration.aerodynamic()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Mars=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Venus=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {satellite: accelerations_settings_satellite}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

# Retrieve the initial state of the satellite using Two-Line-Elements (TLEs)
tle = environment.Tle(line1, line2)
ephemeris = environment.TleEphemeris( "Earth", "J2000", tle, False )
initial_state = ephemeris.cartesian_state( simulation_start_epoch )
#print(element_conversion.cartesian_to_keplerian(initial_state, bodies.get("Earth").gravitational_parameter))

# Define list of dependent variables to save
dependent_variables_to_save = [
    propagation_setup.dependent_variable.altitude(satellite, "Earth")
]

# Create termination settings
termination_time = propagation_setup.propagator.time_termination(simulation_end_epoch)
termination_altitude = propagation_setup.propagator.dependent_variable_termination(propagation_setup.dependent_variable.altitude(satellite, "Earth"), 100E+3, True) #Number is altitude to stop propagating at [m]
hybrid_termination_condition = propagation_setup.propagator.hybrid_termination([termination_time, termination_altitude], True)

# Create numerical integrator settings
#integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(time_step=fixed_step_size, coefficient_set=propagation_setup.integrator.rkf_78)
#integrator_settings = propagation_setup.integrator.bulirsch_stoer_variable_step(initial_time_step=fixed_step_size,extrapolation_sequence = propagation_setup.integrator.deufelhard_sequence, maximum_number_of_steps=7, 
#                                                                                step_size_control_settings =propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-10, 1.0E-10, minimum_factor_increase=0.05),
#                                                                                step_size_validation_settings =propagation_setup.integrator.step_size_validation(0.1, 10000.0),
#                                                                                assess_termination_on_minor_steps = False)
integrator_used = "Runge-Kutta 78, fixed time step" # For text file

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    hybrid_termination_condition,
    output_variables=dependent_variables_to_save
)

# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state and depedent variable history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.dependent_variable_history
dep_vars_array = result2array(dep_vars)

# Plot altitude as function of time and save it
time = (dep_vars_array[:,0] - datetime_to_tudat(date1).epoch()) / (3600 * 24) #In days
EOL_estimate = time[-1]
EOL_date = datetime_to_python(date_time_from_epoch(EOL_estimate * (3600 * 24) + datetime_to_tudat(date1).epoch())).date()
now = datetime.now()
print(f"Final remaining lifetime estimate: {EOL_estimate} days. This estimates re-entry on {EOL_date}")
altitude = dep_vars_array[:, 1] / 1000
plt.figure(figsize=(9, 5))
plt.title(f"{satellite} altitude, starting from {date1.date()}. step size = {fixed_step_size}")
plt.plot(time, altitude)
plt.xlabel('Time [days]')
plt.ylabel('Altitude [km]')
plt.xlim([min(time), max(time)])
plt.grid()
plt.tight_layout()
plt.savefig(f"CURRENT_PREDICTIONS/Altitude over the course of propagation of {satellite} - {now.day}-{now.month}-{now.year} {now.hour}h{now.minute}.png")

# Create text file with all inputs and outputs
ff = open(f"CURRENT_PREDICTIONS\{satellite} EOL Prediction - {now.day}-{now.month}-{now.year} {now.hour}h{now.minute}", "w")
ff.write(f"============== End-of-Life Prediction for {satellite} ==============\n")
ff.write(f"### File creation on {datetime.now()} \n")
ff.write(f"### Associated graph: Altitude over the course of propagation of {satellite} - {now.day}-{now.month}-{now.year} {now.hour}h{now.minute} \n")
ff.write("\n")
ff.write("============================ INPUTS ============================ \n")
ff.write(f"Satellite name:                    {satellite} \n")
ff.write(f"NORAD Catalog ID:                  {satellite_norad_cat_id} \n")
ff.write(f"Propagation starting date:         {date1.date()} \n")
ff.write(f"Satellite mass:                    {satellite_mass} [kg] \n")
ff.write(f"Aerodynamic reference area:        {reference_area} [m^2] \n")
ff.write(f"Drag coefficient:                  {drag_coefficient} \n")
ff.write(f"Radiation Pressure reference area: {reference_area_radiation} [m^2] \n")
ff.write(f"Radiation pressure coefficient:    {radiation_pressure_coefficient} \n")
ff.write("\n")
ff.write(f"TLE: {line1} \n")
ff.write(f"     {line2} \n")
ff.write("\n")
ff.write("=========================== SETTINGS =========================== \n")
ff.write(f"Integrator used:     {integrator_used} \n")
ff.write(f"(Initial) time step: {fixed_step_size} [s] \n")
ff.write(f"Atmospheric model used: {atmomodel}")
ff.write("\n")
ff.write("=========================== OUTPUTS =========================== \n")
ff.write(f"Remaining lifetime estimate: {EOL_estimate} [days] \n")
ff.write(f"                             {EOL_estimate/365} [years] \n")
ff.write(f"Estimated re-entry:          {EOL_date}")

# Play sound to notify of code being finished running
winsound.Beep(440, 1000)