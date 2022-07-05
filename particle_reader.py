#from Junkai. Reads particles from HDF subhalo catalogues. I have adapted slightly. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ellipsoid import EllipsoidTool
from TNG_Galaxy_shape import TNG_Galaxy_shape
from Galaxy_type_SFR import Galaxy_type_SFR
from Illustris_TNG_Galaxy import Illustris_TNG_Galaxy
import illustris_python as il
from http_get import get
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import random
import hdf_reader as hdf
from astropy.io import ascii
dir_TNG='/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/'
my_dir=dir_TNG+'/bulgegrowth/output'
plot_dir=dir_TNG+'/bulgegrowth/plots/'
my_imgs='/home/AstroPhysics-Shared/Sharma/Illustris/images/'


def z_to_snap(z, simulation='TNG100-1', snap=[]):
    #basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/'+simulation+'/output'
    #baseUrl = 'http://www.tng-project.org/api/'
    if len(snap)==0:
       baseUrl = 'http://www.tng-project.org/api/'
       r = get(baseUrl)
       names = [sim['name'] for sim in r['simulations']]
       i = names.index(simulation)
       sim = get( r['simulations'][i]['url'] )
       snap = get( sim['snapshots'] )

    select = 0
    for i in range(len(snap)):
        if abs(snap[i]['redshift']-z)<=0.1 and abs(snap[i]['redshift']-z)<abs(snap[select]['redshift']-z):
           select = i
    if abs(snap[select]['redshift']-z)>0.1:
       print('wrong redshift input')
       return

    snap_ID = snap[select]['number']
    return snap_ID


def SnapNum_to_z(snapnum):
#snap to redshift array. first column is snap, second is redshift, 3rd is scale factor
    snap  = ascii.read('/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output/TNG100-1_snap.lis')
#print(snap)
    snap_array=np.array([snap['num'],snap['redshift']])
    #print(snap_array.shape)
    #print(snap_array[0][0])
    for i in range(len(snap_array[0])):
        if snap_array[0][i]==snapnum:
            z=snap_array[1][i]
        else:
            continue
    return(z)    
#print(snap_array)
#print(snap_array.shape)
#end SnapNum to z-----------------------------------------------------

def scalefactor_to_z(scalefactor):
    #snap to redshift array. first column is snap, second is redshift, 3rd is scale factor
    snap  = ascii.read('/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output/TNG100-1_snap.lis')
#print(snap)
    snap_array=np.array([snap['num'],snap['redshift'],snap['a']])
    #print(snap_array.shape)
    #print(snap_array[0][0])
    for i in range(len(snap_array[2])):
        if snap_array[2][i]>=scalefactor:
            z=snap_array[1][i]
            break
        else:
            continue
    return(z)    
     
 

def Universe_age(redshift, H0=70, Om0=0.3, Tcmb0=2.725):
    cosmo = FlatLambdaCDM(H0, Om0, Tcmb0)
    age   = cosmo.age(redshift)  # units: Gyr
    return age.value
#-- end Universe_age

def Galaxy_Stars(z, Galaxy_ID, hubble=0.7, simulation='TNG100-1', snap=[]):

    snap_ID = z_to_snap(z, simulation='TNG100-1', snap=snap)
    

    basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/'+simulation+'/output'
    headers=il.groupcat.loadHeader(basePath,snap_ID)
    print(headers)

    fields = ['Coordinates', 'Masses', 'GFM_StellarFormationTime','ParticleIDs']

    stars = il.snapshot.loadSubhalo(basePath, snap_ID, Galaxy_ID, 'stars', fields = fields)
    #print(stars.shape())
    #print(stars)

    dum = len(stars['Coordinates'])

    #GFM_StellarFormationTime>0:stars, <=0:wind phase gas cell
    select = np.where(stars['GFM_StellarFormationTime']>0)
    #print('selected stars',select)

    stars_Coord = stars['Coordinates'][select]/hubble/(1+z)
    stars_Mass = stars['Masses'][select]/hubble*10**10
    stars_SFT=stars['GFM_StellarFormationTime'][select]
    star_IDs=stars['ParticleIDs'][select]
    #print(stars_SFT.shape())
    print('star formation time',stars_SFT)
    print('max SFT',np.max(stars_SFT))
    print('min SFT',np.min(stars_SFT))
    print('max coord',np.max(stars_Coord))
    print('min coord',np.min(stars_Coord))
    
    
    print('coordinates before selection',stars_Coord)

    L_box = 75000.0/hubble/(1+z)

    for i in range(3):
        #transposing coord gives 3 columsn one in each direction 
        x_max = max(stars_Coord.T[i]) - min(stars_Coord.T[i])
        print('xmax is',x_max)
        if x_max > 0.5*L_box:
           select = np.where(stars_Coord.T[i] > 0.5*L_box)
           stars_Coord.T[i][select] = stars_Coord.T[i][select] - L_box
           print('I think these are teh dimensions',max(stars_Coord.T[i]) - min(stars_Coord.T[i]), L_box)
    return stars_Coord, stars_Mass, stars_SFT, star_IDs

def calc_Mass_tensor(Coord, Mass, w=[]):

    Total_Mass = np.sum(Mass)
    if len(w) == 0 or len(w)!=len(Mass):
       M = np.dot(Coord.T, Coord*np.array([Mass, Mass, Mass]).T)/Total_Mass
    else:
       M = np.dot(Coord.T, Coord*np.array([Mass/w, Mass/w, Mass/w]).T)/Total_Mass  
    return M 

def calc_Mass_center(Coord, Mass):
    xm = np.sum(Coord.T[0]*Mass, axis=0)/np.sum(Mass, axis=0)
    ym = np.sum(Coord.T[1]*Mass, axis=0)/np.sum(Mass, axis=0)
    zm = np.sum(Coord.T[2]*Mass, axis=0)/np.sum(Mass, axis=0)

    return np.array([xm, ym, zm])

def shift_to_Mass_center(Coord, Mass, center=[]):

    if len(center) == 0: 
       center = calc_Mass_center(Coord, Mass)
    Coord = Coord - np.array([center for i in range(len(Coord))])

    return Coord, Mass

def calc_halfmass_radius(Coord, Mass, Mass_center):

    Coord, Mass = shift_to_Mass_center(Coord, Mass, center=Mass_center)

    r2 = np.sum(Coord**2, axis=1)
    index = np.argsort(r2) # sort the particles from inner to outer

    cum_Mass = np.cumsum(Mass[index]) # cumulate the mass from inner to outer
    cum_Mass_fraction = cum_Mass/np.sum(Mass)

    diff = abs(cum_Mass_fraction-0.5)
    select = np.where(diff == min(diff))[0][0]
    
    radius = np.sqrt(r2[select])

    return radius


# calc ellipsoid at radius R, (normally at R_1/2 or 2*R_1/2)
# calc shape at ellipsoid shell with semi-major length a and width=0.4a
# ellipsoid shell is described with three semi-axis length: radii
# and the rotation matrix: rotation
# R_ellipsoid=1. means ellpsoid at R_ellipsoid = half mass radius
def calc_ellipsoid_R(Coord, Mass, radii, rotation, center=[], R_half=0, R_ellipsoid=1., width=0.4, density0=-1, N_sphere_shell=-1):

    # calculate mass center
    if len(center) == 0:
       center = calc_Mass_center(Coord, Mass)

    R = np.linalg.inv(np.diag(radii**2))

    if R_half <= 0:
       R_half = calc_halfmass_radius(Coord, Mass, Mass_center=center)
    
    #if R_ellipsoid == 0: # by default use half-mass radius
    R_ellipsoid = R_ellipsoid # in the unit of R_half


    # rotate the Coordinates so that the Coordinate axises are along the axis of ellipsoid shell
    r1 = np.zeros(len(Mass)) - 1  # record the 'distance' for each stars to mass center
    for i in range(len(Mass)):
        Coord1 = np.dot(Coord[i,:]-center, np.linalg.inv(rotation))
        r1[i] = np.sqrt(np.dot(np.dot(Coord1,R),Coord1.T)) # in the unit of R_half
    
    select = np.where((r1>R_ellipsoid-width*0.5)&(r1<R_ellipsoid+width*0.5))
    N_stars = len(select[0])
    print(np.max(radii), radii[1]/radii[0], radii[2]/radii[0],  R_ellipsoid, width*0.5, max(r1), min(r1), len(select[0]), N_sphere_shell)
    flag_break = False
    V_circ = (1.2**3-0.8**3) # V_shell/V(R_halfmass)
    #print(V_circ*len(Mass))
    density = len(select[0])/(V_circ*np.pi*radii[0]*radii[1]*radii[2])
    #print('Number density of stars in shell:', density)
    if density0 > 0:
       flag_break = (density<density0)
       if flag_break:
          print(len(select[0]), 'out of', len(Mass), 'galaxies in the elliptical shell')
          return center, radii, rotation, density, flag_break, N_stars

    if N_stars<100 or N_stars<0.01*N_sphere_shell:
       flag_break = True
       print(N_stars, 'stars in the elliptical shell')
       return center, radii, rotation, density, flag_break, N_stars

    Coord0, Mass0 = shift_to_Mass_center(Coord[select], Mass[select], center=center)

    M = calc_Mass_tensor(Coord0, Mass0)
    #w = r1[select]**2 # in unit of R_half**2
    #M = calc_Mass_tensor(Coord0, Mass0, w)

    u, s, rotation = np.linalg.svd(M)

    #rotation, s, vh = np.linalg.svd(M)
    radii = np.sqrt(s) # three semi axis length
    radii = radii/radii[0]*R_ellipsoid*R_half # forced the semi-major length to R_ellipsoid*R_half

    return center, radii, rotation, density, flag_break, N_stars
    
def calc_Galaxy_shape_P(Coord, Mass, R_ellipsoid=1., width=0.4, N_iteration=100, accuracy=0.01):

    time1 = time.time()
    # use spherical shell in the first step
    center = calc_Mass_center(Coord, Mass)
    R_half = calc_halfmass_radius(Coord, Mass, center)
    #print(np.median(Coord.T[0]), np.median(Coord.T[1]), np.median(Coord.T[2]))
    #print(np.mean(Coord.T[0]), np.mean(Coord.T[1]), np.mean(Coord.T[2]))
    #print(center)
    #print(R_half, center)
    #max_y = max(Coord.T[1]) - min(Coord.T[1])
    #for i in range(len(Coord.T[1])):
    #    for j in range(len(Coord.T[1])):
    #        if max_y < abs(Coord.T[1][i] - Coord.T[1][j]):
    #           max_y = abs(Coord.T[1][i] - Coord.T[1][j])
    #print('max y diff:', max_y)           
    radii = np.ones(3)*R_half
    rotation = np.diag([1,1,1]) 
    #radii_current = radii
    #rotation_current = rotation
    B = 1.
    C = 1.

    for i in range(N_iteration):
        print('Radii:',radii,'rotation:',rotation)
        if i == 0:
           #center, radii, rotation, density, flag_break, N_stars 
           results = calc_ellipsoid_R(Coord, Mass, radii=radii, rotation=rotation, center=center, R_half=R_half, R_ellipsoid=R_ellipsoid, width=width)
           N_stars0 = results[5]
        else:
           results = calc_ellipsoid_R(Coord, Mass, radii=radii, rotation=rotation, center=center, R_half=R_half, R_ellipsoid=R_ellipsoid, width=width, density0=density0, N_sphere_shell=N_stars0)
           #density0 = results[3]
        
        flag_break = results[4]
        if flag_break:
           break 
        
        radii_current = results[1] # [a, b, c]
        rotation_current = results[2]
        density0 = results[3]
        N_stars = results[5]

        #radii_order = np.sort(radii_current) # sort from low to high
        B_current = radii_current[1]/radii_current[0]
        C_current = radii_current[2]/radii_current[0]
        
        if ((B-B_current)/max(1e-5, B)<accuracy and (C-C_current)/max(1e-5, C)<accuracy):
           break
        B = B_current
        C = C_current
        radii = radii_current
        rotation = rotation_current

    time2 = time.time()
    print('Calculating intrinsic shape of galaxy with Pillepich2019 method took {0:.1f} seconds.'.format(time2-time1))
    
    return center, radii, rotation, flag_break


def calc_Galaxy_3D_shape(z=0.15, simulation='TNG100-1', width=0.4, R_ellipsoid=1., table_index=0, Ngal=10000, snap=[]):
   
   if len(snap) == 0: 
      baseUrl = 'http://www.tng-project.org/api/'
      r = get(baseUrl)
      names = [sim['name'] for sim in r['simulations']]
      i = names.index(simulation)
      sim = get( r['simulations'][i]['url'] )
      snap = get( sim['snapshots'] )

   snap_ID, ID, M, hubble = TNG_Galaxy_shape(simulation=simulation, z=z)

   print(snap_ID)
   print(len(ID), 'galaxies in total')

   start = table_index*Ngal
   end = min((table_index+1)*Ngal, len(ID))
   #start = 0
   #end = len(ID)
   print('from ID',start,'to ID',end)

   TNG_B = []
   TNG_C = []
   P_B = []
   P_C = []
   reliable = []

   for index in np.arange(start, end):
   #select = np.where(ID==399545)[0][0]
   #for index in [select]:

       TNG_shape = M[index]
       a = TNG_shape[2]
       b = TNG_shape[1]
       c = TNG_shape[0]
       #print('abc', a,b,c)

       stars_Coord, stars_Mass = Galaxy_Stars(z, ID[index], hubble=hubble, simulation=simulation, snap=snap)

       print('Galaxy:',index, ', Nstars:', len(stars_Mass))

       Mass_center = calc_Mass_center(stars_Coord, stars_Mass)
       R_half = calc_halfmass_radius(stars_Coord, stars_Mass, Mass_center)
       print(R_half)
       radii = np.ones(3)*R_half
       rotation = np.diag([1,1,1])
       
       #width = 0.4
       #R_ellipsoid=1.
       center, radii, rotation, flag_break = calc_Galaxy_shape_P(stars_Coord, stars_Mass, R_ellipsoid=R_ellipsoid, width=width, N_iteration=100, accuracy=0.01)

       B = radii[1]/radii[0]
       C = radii[2]/radii[0]
       print('b/a:', b/a, B)
       print('c/a:', c/a, C)

       TNG_B.append(b/a)
       TNG_C.append(c/a)
       P_B.append(B)
       P_C.append(C)
       reliable.append(not flag_break)

   table = pd.DataFrame({'ID':ID[(range(start, end),)],
                         'TNG_B':TNG_B,
                         'TNG_C':TNG_C,
                         'P_B':P_B,
                         'P_C':P_C,
                         'reliable':reliable})

   print('save to file:', 'Galaxy_3D_shape_P2019_'+simulation+'_'+'{0:.1f}'.format(R_ellipsoid)+'Rhalf_z_'+str(z)+'.h5')
   table.to_hdf('Galaxy_3D_shape_P2019_'+simulation+'_'+'{0:.1f}'.format(R_ellipsoid)+'Rhalf_z_'+str(z)+'.h5', 'table_'+str(table_index))


   #plot the star coords. Make sure they are adjusted for centre of mass

def plot_xyz(stars_Coord):
      
   fig, ax = plt.subplots(3,1,figsize=(6,15))
   
   ax[0].plot(stars_Coord.T[0], stars_Coord.T[1], '.', markersize=1)
   ax[0].set_xlabel('x[kpc]')
   ax[0].set_ylabel('y[kpc]')

   ax[1].plot(stars_Coord.T[0], stars_Coord.T[2], '.', markersize=1)
   ax[1].set_xlabel('x[kpc]')
   ax[1].set_ylabel('z[kpc]')

   ax[2].plot(stars_Coord.T[1], stars_Coord.T[2], '.', markersize=1)
   ax[2].set_xlabel('y[kpc]')
   ax[2].set_ylabel('z[kpc]')
   plt.savefig(plot_dir+'Galaxy_Coord_v1.png')
   plt.show()
   plt.close()


def get_3d_plot(x,y,z,xlabel,ylabel,zlabel,legend):
    #remeber to adjust coords  for centre of mass
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    ax.scatter(x,y,z,marker='.',c='k',label=legend)
    origin=0
    ax.scatter(origin,origin,origin,marker='o',c='r',s=4,label='Galactic CoM')
    ax.grid(False)
    plt.xlim(max(x),min(x))
    plt.xlim(max(y),min(y))
    plt.xlim(max(z),min(z))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
   # ax.set_title(plot_title)
    plt.legend()
    plt.show()
    #return(3d_descendant_plot)

   
if __name__ == '__main__':
    redshift=SnapNum_to_z(42)
    stars=Galaxy_Stars(redshift,6,hubble=0.7,simulation='TNG100-1', snap=[])
    sfts=stars[2]
    coords=stars[0]
    print(len(coords))
    mass=stars[1]
    all_IDs=stars[3]
    all_ages=([])
    #center=calc_Mass_center(coords, mass)
    shifted_coords=shift_to_Mass_center(coords, mass, center=[])[0]
    print(len(shifted_coords.T[0]))
    for i in range(len(sfts)):
        #print(sfts[i])
        redshift=scalefactor_to_z(sfts[i])
        age=Universe_age(redshift, H0=70, Om0=0.3, Tcmb0=2.725)
        all_ages=np.append(all_ages,age)
    #print(all_ages)
    print(np.max(all_ages))
    print(np.min(all_ages))
    idx=np.where(all_ages<.5)
    #print(idx)
    selected_stars=shifted_coords[idx]
    selected_IDs=all_IDs[idx]
    x=selected_stars.T[0]
    y=selected_stars.T[1]
    z=selected_stars.T[2]
    #print(selected_stars)
    get_3d_plot(x,y,z,'x','y','z','stars less than 0.5Gyr stars')    
       
   #stars_Coord, stars_Mass = Galaxy_Stars(z, Galaxy_ID, hubble=hubble, simulation=simulation, snap=snap) 

   #xm = np.sum(stars_Coord.T[0]*stars_Mass, axis=0)/np.sum(stars_Mass, axis=0)
   #ym = np.sum(stars_Coord.T[1]*stars_Mass, axis=0)/np.sum(stars_Mass, axis=0)
   #zm = np.sum(stars_Coord.T[2]*stars_Mass, axis=0)/np.sum(stars_Mass, axis=0)
   
   #Coord = stars_Coord - np.array([[xm, ym, zm] for i in range(len(stars_Mass))])

   #M = calc_Mass_tensor(Coord, stars_Mass)
   #print(M)

   #calc_3D_shape_P2019(Coord, stars_Mass)

   #radius = calc_halfmass_radius(Coord, stars_Mass)
   #print(radius)

   #R_half = calc_halfmass_radius(Coord, stars_Mass)
   #radii = np.ones(3)*R_half
   #rotation = np.diag([1,1,1])
   #center, radii, rotation = calc_ellipsoid_R(Coord, stars_Mass, radii, rotation, R_ellipsoid=1., width=0.4)
   
   #print(center, radii, rotation)

   #center, radii, rotation = calc_Galaxy_shape_P(Coord, stars_Mass, R_ellipsoid=1., width=0.4, N_iteration=100, accuracy=0.01)

   #print(center, radii, rotation)

   #plot_xyz(z=0.44, Galaxy_ID=399545)



