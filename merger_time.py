#from Junkai and Stijns code
import illustris_python as il
import numpy as np
from http_get import get
import h5py
import os
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

def Universe_age(redshift, H0=70, Om0=0.3, Tcmb0=2.725):
    cosmo = FlatLambdaCDM(H0, Om0, Tcmb0)
    age = cosmo.age(redshift)
    #print(age.value)
    return age.value

def snapNum_Z(snapNum, sim_name='TNG100-1', snap_list=None):

    if not snap_list:
       baseUrl = 'http://www.tng-project.org/api/'
       r = get(baseUrl)

       names = [sim['name'] for sim in r['simulations']]
       i = names.index(sim_name)
       sim = get( r['simulations'][i]['url'] )

       snap_list = get( sim['snapshots'] )
    
    return snap_list[snapNum]['redshift']

def read_tree(snapNum, subfindID, sim_name='TNG100-1'):
    basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'
    fields = ['SubfindID', 'SnapNum', 'SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID', 'SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloSFRinRad']
    tree = il.sublink.loadTree(basePath, snapNum, subfindID, fields=fields)
    tree_main = il.sublink.loadTree(basePath, snapNum, subfindID, fields=fields, onlyMPB=True) 

    stellar_mass = tree['SubhaloMassInRadType'].T[4]
    hubble = 0.7
    select = np.where(stellar_mass>0)
    lmass = -99 + np.zeros(len(stellar_mass))
    lmass[select] = np.log10(stellar_mass[select]/hubble) + 10.
    
    return tree, tree_main, lmass

def get_merger_tlookback(snapNum, subfindID, massratio_range=[0.333, 1.], sim_name='TNG100-1', snap_list=None):

    if not snap_list:
       baseUrl = 'http://www.tng-project.org/api/'
       r = get(baseUrl)

       names = [sim['name'] for sim in r['simulations']]
       i = names.index(sim_name)
       sim = get( r['simulations'][i]['url'] )

       snap_list = get( sim['snapshots'] )


    tree, tree_main, lmass = read_tree(snapNum, subfindID, sim_name=sim_name)
    #print('SnapNum:', tree_main['SnapNum']) 
    #print('SubfindID:', tree_main['SubfindID'])
    #print('SubhaloID:', tree_main['SubhaloID'])
    #print('FirstProgenitorID:', tree_main['FirstProgenitorID'])
    #print('NextProgenitorID:', tree_main['NextProgenitorID'])
    #for i in np.arange(max(tree['SnapNum']), min(tree['SnapNum']), -1):
    SubhaloID = tree_main['SubhaloID']
    SnapNum = tree_main['SnapNum']

    snap = []#[[] for i in range(len(SnapNum))]
    redshift = []
    mass_ratio = []#[[] for i in range(len(SnapNum))]
    #print(len(SnapNum))
    for i in range(len(SnapNum)):
        #print(i)
        index = np.where(tree['SubhaloID']==SubhaloID[i])[0][0]
        #print('SnapNum', SnapNum[i], 'SubhaloID:', SubhaloID[i], 'lmass:', lmass[index])
        if tree['FirstProgenitorID'][index] == -1:
           continue 
        First_index = np.where(tree['SubhaloID']==tree['FirstProgenitorID'][index])[0][0]
        
        lmass_first = lmass[First_index]
        if lmass_first < 0:
           break

        #print('FirstProgenitor, SubhaloID:', tree['FirstProgenitorID'][index], 'SnapNum:', tree['SnapNum'][First_index], 'lmass:', lmass[First_index])

        Next = tree['NextProgenitorID'][First_index]
        lmass_next = []
        snap_next = []
        while Next != -1:
              Next_index = np.where(tree['SubhaloID']==Next)[0][0]
              if lmass[Next_index] >0:
                 #print('NextProgenitor, SubhaloID:', Next, 'SnapNum:', tree['SnapNum'][Next_index], 'lmass:', lmass[Next_index])
                 lmass_next.append(lmass[Next_index])
                 snap_next.append(tree['SnapNum'][index])  #[Next_index])
              Next = tree['NextProgenitorID'][Next_index]
              
        if len(lmass_next) == 0:
           continue 
        
        for j in range(len(lmass_next)):
            if lmass_next[j] - lmass_first > np.log10(massratio_range[0]) and lmass_next[j] - lmass_first < np.log10(massratio_range[1]):
               snap.append(snap_next[j])
               redshift.append(snapNum_Z(snapNum=snap_next[j], sim_name=sim_name, snap_list=snap_list))
               mass_ratio.append(10**(lmass_next[j] - lmass_first))

    #print(redshift, snap)
        #print('--------------')
    U_age = Universe_age(redshift=0, H0=70, Om0=0.3, Tcmb0=2.725)
    if len(redshift) != 0:
       tlookback =  U_age-Universe_age(redshift=redshift, H0=70, Om0=0.3, Tcmb0=2.725) 
    else:
       tlookback = [] 
    #class mergertime:
    #      def __init__(self, snapNum, subfindID, massratio_range, sim_name, tlookback, mass_ratio):
    #          self.snapNum = snapNum
    #          self.subfindID = subfindID
    #          self.massratio_range = massratio_range
    #          self.sim_name = sim_name

    #          self.tlookback = tlookback
    #          self.mass_ratio = mass_ratio

    #mergertime = mergertime(snapNum, subfindID, massratio_range, sim_name, np.array(tlookback)*u.Gyr, np.array(mass_ratio))
    
    mergertime = {'snapNum':snapNum, 'subfindID':subfindID, 'massratio_range':massratio_range , 'sim_name':sim_name, 'tlookback':np.array(tlookback)*u.Gyr, 'mass_ratio':np.array(mass_ratio)}
    #print(mergertime.lookback)
    #print(mergertime.mass_ratio)
    #print(mergertime.snapNum)
    print('merger lookback times:',mergertime['tlookback'])    
    return mergertime
                


if __name__ == '__main__':
   #SnapNum_Z(99, sim_name='TNG100-1') 
   #Universe_age(redshift=3, H0=70, Om0=0.3, Tcmb0=2.725)
   sim_name = 'TNG100-1'
   baseUrl = 'http://www.tng-project.org/api/'
   r = get(baseUrl)

   names = [sim['name'] for sim in r['simulations']]
   i = names.index(sim_name)
   sim = get( r['simulations'][i]['url'] )

   snap_list = get( sim['snapshots'] )
   
   U_age = Universe_age(redshift=0, H0=70, Om0=0.3, Tcmb0=2.725)
   print('Universe age:', U_age*u.Gyr)
   for i in range(20):
       get_merger_tlookback(snapNum=99, subfindID=i, massratio_range=[0.333, 1.], sim_name=sim_name, snap_list=snap_list)


