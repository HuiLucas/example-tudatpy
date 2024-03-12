# First time running code in this repo: activate conda environment and run: conda create --name re-entrypredictionenv --file requirements.txt

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

# Define the login data
login_data = {
    'identity': 'wschaerlaecken@gmail.com',
    'password': 'groupd03123456789'}

NORAD_CAT_ID = 32789 #Delfi C3
DATE = "2008-04-27--2008-04-29"
# Create a session
with requests.Session() as session:
    # Post the login data
    post_response = session.post('https://www.space-track.org/ajaxauth/login', data=login_data)

    # Check if login was successful
    if post_response.status_code == 200:
        # If login is successful, make the GET request
        url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{NORAD_CAT_ID}/orderby/EPOCH DESC/EPOCH/{DATE}/format/tle"
        print(url)
        get_response = session.get(url)

        if get_response.status_code == 200:
            data = get_response.text
            print(data)
        else:
            print("Failed to retrieve data. Status code:", get_response.status_code)
            print("Response text:", get_response.text)
    else:
        print("Failed to log in. Status code:", post_response.status_code)
        print("Response text:", post_response.text)

        # Split the data into individual lines
    lines = data.split('\n')

        # Iterate over each line and assign the values to variables
for line in lines:
    if line.startswith('1 '):
        line1 = line.split(' ')
    elif line.startswith('2 '):
        line2 = line.split(' ')

for line in lines:
    if line.startswith('1 '):
        line1 = re.findall(r'\d+\.\d+|\d+', line)  # Find all numeric characters and decimal points
    elif line.startswith('2 '):
        line2 = re.findall(r'\d+\.\d+|\d+', line)  # Find all numeric characters and decimal points

# Print the assigned values
print("Line 1:", line1)
print("Line 2:", line2)

Sat_num = line1[1]
Int_Des_Year=line1[2][:2]
Int_Des = line1[2][2:]
Epoch_Year = line1[3][:2]
Epoch_Day = line1[3][2:]
B = int(line1[4])*10**(-8)
Second_Der_Mean_Motion = (float(line1[5])/100000) * 10**(-int(line1[6]))
BSTAR = int(line1[7])*10**(-5) * 10**(-int(line1[8]))
Ephemeris = line1[9]
Element_Number = line1[10][:3]
Check_Sum_1 = line1[10][3]

Inclination = line2[2]
RAAN = line2[3]
Eccentricity = int(line2[4])*10**(-7)
Arg_Perigee = line2[5]
Mean_Anomaly = line2[6]
Mean_Motion = line2[7][:11]
Rev_Num = line2[7][11:16]
Check_Sum_2 = line2[7][16:]

# Calculate the launch year
if int(Int_Des_Year) < 57:
    launch_year = int(Int_Des_Year)+2000
else:
    launch_year = int(Int_Des_Year)+1900

# Calculate the period and semi-major axis
Period = (1*24*3600)/(float(Mean_Motion))
semi_major_axis = (Period**2 * 3.9860044188*10**14/((2*math.pi)**2))**(1/3)
rp = semi_major_axis*(1-Eccentricity)
ra = semi_major_axis*(1+Eccentricity)

Height_apo = ra/1000 - 6371
Height_peri = rp/1000 - 6371

#.calculating the velocity
Vp = math.sqrt((3.9860044188*10**14*2*ra)/(rp*(ra+rp)))
Va = math.sqrt((3.9860044188*10**14*2*rp)/(ra*(ra+rp)))

# Convert the epoch year and day to an actual date and time
if int(Epoch_Year) < 30:
    epoch_year = int(Epoch_Year) + 2000  # Add 2000 to get the full year
else:
    epoch_year = int(Epoch_Year) + 1900  
start_of_year = datetime(epoch_year, 1, 1)  # January 1 of the epoch year
epoch_day = float(Epoch_Day)  # Convert to float to handle fractional days

# Subtract 1 because the epoch day is 1-indexed, but timedelta days are 0-indexed
actual_date = start_of_year + timedelta(days=epoch_day - 1)

print("Satellite number:", Sat_num)
print("International Designator Year:", Int_Des_Year)
print("International Designator:", Int_Des)
print("Launch Year:", launch_year)
print("Epoch Year:", Epoch_Year)
print("Epoch Day:", Epoch_Day)
print("B:", B)
print("Second Derivative of Mean Motion:", Second_Der_Mean_Motion)
print("BSTAR:", BSTAR)
print("Ballistic Coefficient from BSTAR:", 2*BSTAR/(0.15696615))
print("Ephemeris:", Ephemeris)
print("Element Number:", Element_Number)
print("Check Sum 1:", Check_Sum_1)
print("Inclination:", Inclination)
print("RAAN:", RAAN)
print("Eccentricity:", Eccentricity)
print("Arg Perigee:", Arg_Perigee)
print("Mean Anomaly:", Mean_Anomaly)
print("Mean Motion:", Mean_Motion)
print("Rev Num:", Rev_Num)
print("Check Sum 2:", Check_Sum_2)

print("Actual date and time:", actual_date)
print("Period:", Period/60, "Minutes")
print("Semi Major Axis:", semi_major_axis/1000, "km")
print("Perigee:", rp/1000, "km")
print("Apogee:", ra/1000, "km")
print("Height Apogee:", Height_apo, "km")
print("Height Perigee:", Height_peri, "km")
print("Velocity at Perigee:", Vp*3.6, "km/h")
print("Velocity at Apogee:", Va*3.6, "km/h")

# Create a new figure with a map projection
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

plt.title(f'Orbit on {actual_date} of satellite {Sat_num}')
# Generate an array of angles from 0 to 2*pi
theta = np.linspace(0, 2*np.pi, 1000)


# Calculate the semi-minor axis
semi_minor_axis = semi_major_axis * np.sqrt(1 - Eccentricity**2)

# Calculate the x, y, and z coordinates of the orbit
x = semi_major_axis * np.cos(theta) - semi_major_axis * Eccentricity
y = semi_minor_axis * np.sin(theta)
z = np.zeros_like(x)

# Convert the argument of perigee, inclination, and longitude of ascending node to radians
arg_perigee_rad = np.radians(float(Arg_Perigee))
inclination_rad = np.radians(float(Inclination))
lon_asc_node_rad = np.radians(float(RAAN))

# Define the rotation matrices
rotation_z1 = np.array([[np.cos(lon_asc_node_rad), -np.sin(lon_asc_node_rad), 0],
                        [np.sin(lon_asc_node_rad), np.cos(lon_asc_node_rad), 0],
                        [0, 0, 1]])
rotation_x = np.array([[1, 0, 0],
                       [0, np.cos(inclination_rad), -np.sin(inclination_rad)],
                       [0, np.sin(inclination_rad), np.cos(inclination_rad)]])
rotation_z2 = np.array([[np.cos(arg_perigee_rad), -np.sin(arg_perigee_rad), 0],
                        [np.sin(arg_perigee_rad), np.cos(arg_perigee_rad), 0],
                        [0, 0, 1]])

# Apply the rotations
orbit_xyz = np.vstack((x, y, z))
rotated_orbit_xyz = rotation_z1 @ rotation_x @ rotation_z2 @ orbit_xyz

# Extract the rotated x, y, and z coordinates
x_rotated, y_rotated, z_rotated = rotated_orbit_xyz

# Convert Cartesian coordinates to latitude and longitude
r = np.sqrt(x_rotated**2 + y_rotated**2 + z_rotated**2)
lat = np.arcsin(z_rotated / r) * 180 / np.pi  # Latitude in degrees
lon = np.arctan2(y_rotated, x_rotated) * 180 / np.pi  # Longitude in degrees

# Plot the orbit on the map
ax.plot(lon, lat, transform=ccrs.Geodetic())

# Set the extent of the map
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

# Show the plot
#plt.show()



def get_tle_data(norad_cat_id, dates):
    # Function to implement with eol prediction
    # INPUTS: norad_cat_id = int, date = str
    # OUTPUTS: semi major axis[km], eccentricity[/], inclination[deg], argument of periapsis[deg], 
    #          right ascension of the ascending node[deg], true anomaly[deg]
    # EXAMPLE: get_tle_data(51074, "2022-09-06--2022-09-07")
    import requests
    import re
    import math

    # Define the login data
    login_data = {
        'identity': 'wschaerlaecken@gmail.com',
        'password': 'groupd03123456789'}

    NORAD_CAT_ID = norad_cat_id
    DATE = dates
    # Create a session
    with requests.Session() as session:
        # Post the login data
        post_response = session.post('https://www.space-track.org/ajaxauth/login', data=login_data)

        # Check if login was successful
        if post_response.status_code == 200:
            # If login is successful, make the GET request
            url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{NORAD_CAT_ID}/orderby/TLE_LINE1%20ASC/EPOCH/{DATE}/format/tle"
            print(url)
            get_response = session.get(url)

            if get_response.status_code == 200:
                data = get_response.text
                print(data)
            else:
                print("Failed to retrieve data. Status code:", get_response.status_code)
                print("Response text:", get_response.text)
        else:
            print("Failed to log in. Status code:", post_response.status_code)
            print("Response text:", post_response.text)

            # Split the data into individual lines
        lines = data.split('\n')

            # Iterate over each line and assign the values to variables
    for line in lines:
        if line.startswith('1 '):
            line1 = line.split(' ')
        elif line.startswith('2 '):
            line2 = line.split(' ')

    for line in lines:
        if line.startswith('1 '):
            line1 = re.findall(r'\d+\.\d+|\d+', line)  # Find all numeric characters and decimal points
        elif line.startswith('2 '):
            line2 = re.findall(r'\d+\.\d+|\d+', line)  # Find all numeric characters and decimal points

    Sat_num = line1[1]
    Int_Des_Year=line1[2][:2]
    Int_Des = line1[2][2:]
    Epoch_Year = line1[3][:2]
    Epoch_Day = line1[3][2:]
    B = int(line1[4])*10**(-8)
    Second_Der_Mean_Motion = (float(line1[5])/100000) * 10**(-int(line1[6]))
    BSTAR = int(line1[7])*10**(-5) * 10**(-int(line1[8]))
    Ephemeris = line1[9]
    Element_Number = line1[10][:3]
    Check_Sum_1 = line1[10][3]

    Inclination = line2[2]
    RAAN = line2[3]
    Eccentricity = int(line2[4])*10**(-7)
    Arg_Perigee = line2[5]
    Mean_Anomaly = line2[6]
    Mean_Motion = line2[7][:11]
    Rev_Num = line2[7][11:16]
    Check_Sum_2 = line2[7][16:]


    # Calculate the period and semi-major axis
    Period = (1*24*3600)/(float(Mean_Motion))
    semi_major_axis = (Period**2 * 3.9860044188*10**14/((2*math.pi)**2))**(1/3)
    rp = semi_major_axis*(1-Eccentricity)
    ra = semi_major_axis*(1+Eccentricity)

    # Calculate eccentric anomaly
    mean_anomaly_rad = Mean_Anomaly * math.pi / 180
    Eccentricity_rad = Eccentricity * math.pi / 180
    converged = False
    E_old = mean_anomaly_rad
    while converged == False:
        E_new = mean_anomaly_rad + Eccentricity * math.sin(E_old)
        if math.abs((E_new - E_old)/E_new) < 0.01:
            converged = True
        E_old = E_new
    eccentric_anomaly = E_old
    true_anomaly_rad = math.acos((math.cos(eccentric_anomaly)-Eccentricity_rad) / (1 - Eccentricity_rad * math.cos(eccentric_anomaly)))
    true_anomaly = true_anomaly_rad * 180 / math.pi

    return semi_major_axis, Eccentricity, Inclination, Arg_Perigee, RAAN, true_anomaly

"""
semi_major_axis=7500.0e3,
eccentricity=0.1,
inclination=np.deg2rad(85.3),
argument_of_periapsis=np.deg2rad(235.7),
longitude_of_ascending_node=np.deg2rad(23.4),
true_anomaly=np.deg2rad(139.87)
"""