
##This code holds functions that allow pulling of a defined catalogue from the TNGIllustris database and images from the  visualiser. It also contains short scripts that plot and allow analysis of data. based on Stijns TNG Mk sample code and built on by RS. import numpy sysimport3        as np
import os
import illustris_python as il
import pandas as  pd
from http_get import get
import h5py
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
import random
import hdf_reader as hdf
import merger_tree as mt
from photutils.aperture import EllipticalAperture,CircularAperture,CircularAnnulus,BoundingBox,aperture_photometry,ApertureMask
import seaborn as sns
import deepdish as dd
np.set_printoptions(threshold=np.inf)

dir_TNG='/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/'
my_dir='/home/AstroPhysics-Shared/Sharma/Illustris/data_output/'
plot_dir='/home/AstroPhysics-Shared/Sharma/Illustris/plots/'
my_imgs='/home/AstroPhysics-Shared/Sharma/Illustris/images/'
output_dir=dir_TNG+'bulgegrowth/output/'
list_dir=dir_TNG+'bulgegrowth/lists/'


def SnapNum_to_z(snapnum):
#snap to redshift array. first column is snap, second is redshift. Third is scale. Uses snap lis file.  
    snap  = ascii.read('/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output/TNG100-1_snap.lis')
    snap_array=np.array([snap['num'],snap['redshift']])
    for i in range(len(snap_array[0])):
        if snap_array[0][i]==snapnum:
            z=snap_array[1][i]
        else:
            continue
    return(z)    

#end SnapNum to z-----------------------------------------------------


def Universe_age(redshift, H0=70, Om0=0.3, Tcmb0=2.725):
    cosmo = FlatLambdaCDM(H0, Om0, Tcmb0)
    age   = cosmo.age(redshift)  # units: Gyr
    return age.value
#-- end Universe_age

def z_to_SnapNum(z, sim_name='TNG50-1'):
    baseUrl = 'http://www.tng-project.org/api/'
    r = get(baseUrl)

    names = [sim['name'] for sim in r['simulations']]
    i = names.index(sim_name)

    sim  = get( r['simulations'][i]['url'] )
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
#--end z_to_SnapNum------------------------------------------------

#Gets the subhalo ID from the subfind and offsets
def get_SubhaloID(snapNum, SubfindID, sim_name='TNG50-1', verb=False):
    basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/'+sim_name+'/output'
    offsetFile = il.groupcat.offsetPath(basePath, snapNum)
    prefix = 'Subhalo/SubLink/'

    idx_srt = np.argsort(SubfindID)
    srt_SubfindID = list(SubfindID[idx_srt])
    with h5py.File(offsetFile, 'r') as f:
        srt_SubhaloID = f[prefix+'SubhaloID'][srt_SubfindID]
        # indexing with srt_SubfindID needs to be sorted indices

    SubhaloID = np.empty_like(srt_SubhaloID)
    SubhaloID[idx_srt] = srt_SubhaloID
    if verb:
        print("SubfindID ", SubfindID)
        print("SubhaloID ", SubhaloID)
    return SubhaloID


#--end get_SubhaloID---------------------------------------------

#gets the subfind Id from the subhalo ID
def get_SubfindID(snapNum, SubhaloID, sim_name='TNG50-1', verb=False):
    basePath = dir_TNG+sim_name+'/output'
    offsetFile = il.groupcat.offsetPath(basePath, snapNum)
    prefix = 'Subhalo/SubLink/'
                         
    with h5py.File(offsetFile, 'r') as f:
        if len(SubhaloID) == 1:
            SubfindID = np.where(f[prefix+'SubhaloID']==np.array(SubhaloID).reshape(-1)[0])[0]
        elif len(SubhaloID) > 1:
            SubfindID = []
            for SubhaloID_i in SubhaloID:
                SubfindID.append(np.where(np.array(f[prefix+'SubhaloID'])==SubhaloID_i)[0][0])
    SubfindID = np.array(SubfindID)
    if verb:
        print("SubhaloID ", SubhaloID)
        print("SubfindID ", SubfindID)
        
    return SubfindID

#end get _subfind--------------------------------------------------------------------


   

def sel_btbymass(z,mass_min,range_size,sim_name='TNG50-1'):
#select all galaxies by stellar min mass and mass range at any z and  collect their BT ratio,massmSFR,sSFR,radius. Also output snapnum and  subfindIDs 
     basePath='/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/'+sim_name+'/output'
     hubble=0.7
     snapnum=z_to_SnapNum(z)
     age_univ=Universe_age(z)
     mass_max=mass_min+range_size
     print('snapnum is',snapnum)
     print('basePath',basePath)            
     print('now getting subhalos')
     subhalos=il.groupcat.loadSubhalos(basePath,snapnum,fields=['SubhaloMassInRadType','SubhaloSFRinRad','SubhaloHalfmassRadType'])
     lmstar=np.log10(subhalos['SubhaloMassInRadType'][:,4]/hubble)+10
     lSFR=np.log10(np.clip(subhalos['SubhaloSFRinRad'], a_min=1.0e-4, a_max=None))
     lSSFR=lSFR-lmstar
     rad=np.log10(subhalos['SubhaloHalfmassRadType'][:,4]/hubble)+10
     #return the indices of the relavent halos. Those with speciifc SFR greater than -11 also.      
     idx=np.where((mass_min<=lmstar)&(lmstar<=mass_max)&(lSSFR >= (-np.log10(3.*age_univ) - 9)))[0]
     #print(idx)
#return the relevent  masses
     sel_mass=lmstar[idx]
     sel_lSFR=lSFR[idx]
     sel_lSSFR=lSSFR[idx]
     sel_rad=rad[idx]
#get relavent subfind IDs . These have same idx order as mass      
     sel_subhaloID = get_SubhaloID(snapnum, idx)
     sel_subfindID = get_SubfindID(snapnum, sel_subhaloID)
     #print(snapnum,sel_subfindID,sel_mass,lSSFR)
     return(snapnum,sel_subfindID,sel_mass,sel_lSFR,sel_lSSFR,sel_rad)
          
 
#end btbymass---------------- -----------------------------------------------------------------------

def get_circs(subfindID,snapnum,sim_name='TNG50-1'):
#Now get stellar circs from these subfindIDs from stellar circ hdf5 file directly
#pathfile is valid for all TNGs. Can only pass one subfind at a time
     filepath=dir_TNG+sim_name+'/postprocessing/stellar_circs/'
     filename=filepath+'stellar_circs.hdf5'
     hubble=0.7
#the keys in the hdf file are snapshot first     
     Snapshot='Snapshot_'+str(snapnum)
#read the file and the snapshot     
     f=h5py.File(filename,'r')
     Snap=f[Snapshot]
     Circs=Snap['CircTwiceBelow0Frac']
     Circ_ID=Snap['SubfindID']
     for i in range(len(Circ_ID)):
         if Circ_ID[i]==subfindID:
             final_Circs=Circs[i]
     print('Got Circ for Subfind ID')
     return(snapnum,final_Circs,subfindID)

#end get_circs---------------------------------------------------------------------------------- 

   
#uses Junkai's merger tree ID code as my code was not finding all descendants. I adapted it to add some fields. At lower mass <9 some descendants may be missing from the tree.  
def get_descendants(snapnum,ID):
    hubble=0.7
    tree=mt.find_merger_tree(snapnum,ID)
    #returns a dictionary of descendent IDs and subfind IDs as well as mss sfr and rad
    ID=tree.get('SubfindID')
    snap=tree.get('SnapNum')
    mstar=tree.get('SubhaloMassInRadType')
    lmstar=np.log10(mstar[:,4]/hubble)+10
    #print('got lmstars')
    SFR=tree.get('SubhaloSFRinRad')
    lSFR=np.log10(SFR)
    #print(lSFR)
    lSSFR=lSFR-lmstar
    #comoving half mass radius
    rad=tree.get('SubhaloHalfmassRadType')
    lrad=np.log10(rad[:,4]/hubble)+10
    return(ID,snap,lmstar,lrad,lSFR,lSSFR)



#end get cluster index------------------------------------------------


       

    
#=== Execute scripts below after uncommenting. These scripts were used to generate the catalogues and CSVs and data frames===
'''      
#Get images needed
z=SnapNum_to_z(53)
sel_galaxies=np.array(sel_btbymass(z,10,1.5))
init_snapnum=sel_galaxies[0]
print('starting snapnum',init_snapnum)
all_ID=sel_galaxies[1]
missing_IDs=[]
for i in range(len(all_ID)):
    n_left=len(all_ID)-i
    print(f'..........................there are {n_left} left........ ')
    ID=all_ID[i]
    try:
        hdf.get_hdf(init_snapnum,ID,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    except:
        print('Unable to get image')
        missing_IDs.append(ID)
        continue
    print('got image')
print(missing_IDs)
'''

#lst=[62382,169816,286154,324861,325680,325983,327822]
#for i,ID in enumerate(lst):
#    hdf.get_hdf(53,ID,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')

'''
#get dendrogram leaf index CAS and Blob_index and position and Bulge ratio and mass and log SFR and sSFR and stellar half mass radius.Also get galactocentric blob positions normalised against r_pet. output into dataframe and csv. Updated with new clump algorithm with blurring and exclusion of central kpc when calculating clump index. Updated more to get med blob pos adn struct inner and stick in h5 file.    
z=SnapNum_to_z(53)
sel_galaxies=np.array(sel_btbymass(z,10,1.5))
init_snapnum=sel_galaxies[0]
print('starting snapnum',init_snapnum)
all_ID=sel_galaxies[1]
all_clump=np.array([])
all_asym=np.array([])
all_conc=np.array([])
all_leaf=np.array([])
all_blob=np.array([])
all_mass=sel_galaxies[2]
all_circs=np.array([])
all_log_SFR=sel_galaxies[3]
all_log_sSFR=sel_galaxies[4]
all_radius=sel_galaxies[5]
med_rad_pos=np.array([])
Q1_rad_pos=np.array([])
Q3_rad_pos=np.array([])
n_clumps=np.array([])
print('radius length',all_radius.size)
all_blob_pos=[]
#final_values=([])
for i in range(len(all_ID)):
    n_left=len(all_ID)-i
    print('there are left',n_left)
    ID=int(all_ID[i])
    final_circs=get_circs(ID,init_snapnum)
    all_circs=np.append(all_circs,final_circs[1])
    #get images first and then apply functions
    #img=hdf.get_hdf(init_snapnum,ID,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    img=my_imgs+'grid_subhalo_TNG50-1_'+str(init_snapnum)+'_'+str(ID)+'.hdf5'
    img_array=hdf.get_lin_array(img)
    lin_img=img_array[0]
    log_img=img_array[1]
    R_pet=hdf.get_petrosian(log_img,200,200)
    print(f'R pet is {R_pet}')
    cent_1kpc=int(40/all_radius[i])
    blur_rad=int(cent_1kpc*0.637)
    d=hdf.get_dendro(lin_img,R_pet,200,200,blur_rad)
    leaf_index=hdf.get_leaf_flux(d[1],d[0],R_pet)
    blob_img=hdf.get_clumps_img(lin_img,R_pet,200,200,blur_rad)
    blobs=hdf.get_clumps(blob_img[0])
    blob_pos=[]
    for pos in range(len(blobs[:,])):
        positionx=100-blobs[pos,1]
        positiony=100-blobs[pos,0]
        gal_r=np.sqrt(positionx**2+positiony**2)
        #just use clumps that have position >1kpc from centre
        norm_gal_r=gal_r/100
        blob_pos.append(norm_gal_r)
    blob_mask=hdf.get_clump_mask(blob_img[2],blobs,200,200)
    blob_index=hdf.get_clump_light(blob_img[2],blobs,blob_mask,R_pet)
    all_blob=np.append(all_blob,blob_index)
    blob_pos=ID,blob_pos
    blob_pos=tuple(blob_pos)
    print(blob_pos)
    #results in a seperate table with ID as odd idx and position radii as  even idx lists 
    all_blob_pos=np.append(all_blob_pos,blob_pos)
    #print(all_blob_pos)
    Clump_index=hdf.get_clump_index(lin_img,200,200,R_pet,blur_rad)
    Asym_index=hdf.get_asymm(lin_img,200,200)
    #print(Clump_index)
    all_asym=np.append(all_asym,Asym_index)
    Conc_index=hdf.get_CI(lin_img,200,200,R_pet)
    all_conc=np.append(all_conc,Conc_index)
    all_clump=np.append(all_clump,Clump_index)
    all_leaf=np.append(all_leaf,leaf_index)
    #print('all clump is',all_clump,all_clump.shape) 
print(all_clump.size)
print(all_leaf.size)
print(all_asym.size)
print(all_conc.size)
print(all_ID.size)
print(all_mass.size)
#print(final_values.shape)
#output clump radial positions as a list
#now write all blob pos and then open and extract medians and IQR and n
with open(my_dir+'New_blob_pos_snap53','w') as f:
    for item in all_blob_pos:
        f.write('%s \n'%item)
with open(my_dir+'New_blob_pos_snap53','r') as f:
    for (i,line) in enumerate(f):
        #print('line number',i)
        lines=line.rstrip()
        lines=lines.lstrip()
        lines=lines.lstrip('[')
        lines=lines.rstrip(']')
        lines=lines.strip(' ')
        pos=np.array((lines.split(',')))
        if len(pos)==0:
            pos=np.array(float(0))
    #print(pos,pos.shape)
        if i%2==1:
            new_pos=np.array([])
            for i in range(len(pos)):
                pos_r=float(pos[i] or 0)
                new_pos=np.append(new_pos,pos_r)
            #print(new_pos)
            med_pos=np.nanpercentile((new_pos),50)
            Q1_pos=np.nanpercentile(new_pos,25)
            Q3_pos=np.nanpercentile(new_pos,75)
            clump_n=len(pos)
            med_rad_pos=np.append(med_rad_pos,med_pos)
            Q1_rad_pos=np.append(Q1_rad_pos,Q1_pos)
            Q3_rad_pos=np.append(Q3_rad_pos,Q3_pos)
            n_clumps=np.append(n_clumps,clump_n)
        else:
            continue
print(len(med_rad_pos))
IQR=Q3_rad_pos-Q1_rad_pos
ID_list=all_ID
#for name in col_names:
#    print(name)
all_inner_struct=np.array([])
all_total_struct=np.array([])
for i in range(len(ID_list)):
    print(f'---------------------------------------------------i is {i}. There are {len(ID_list)-i} left')
    ID_new=int(ID_list[i])
    img=my_imgs+'grid_subhalo_TNG50-1_'+str(init_snapnum)+'_'+str(ID_new)+'.hdf5'
    img_array=hdf.get_lin_array(img)
    lin_img=img_array[0]
    log_img=2**(img_array[1]-35)
    #log_img=plt.imshow(log_img,cmap='viridis')
    struct=hdf.get_structure(lin_img,r=100,dr=2,theta=180)
    total_struct=np.sum(struct[2])
    inner_struct=np.sum(struct[2][0:20])/np.sum(struct[2])
    all_inner_struct=np.append(all_inner_struct,inner_struct)
    all_total_struct=np.append(all_total_struct,total_struct)
final_values=np.array((all_ID,all_mass,all_circs,all_radius,all_log_SFR,all_log_sSFR,all_clump,all_asym,all_conc,all_blob,all_leaf,med_rad_pos,IQR,n_clumps,all_inner_struct,all_total_struct))
print(final_values.shape)
#print('blob shape',all_blob_pos.shape)
final_values=np.transpose(final_values)
print(final_values.shape)
#all_blob_pos=np.transpose(all_blob_pos)
#print(all_blob_pos)
#final_pos_data=pd.DataFrame(all_blob_pos,columns=['ID','gal_cent_pos'])
final_data=pd.DataFrame(final_values,columns=['ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx','Med_clump_pos','Clump_pos_IQR','N_clumps','Inner_m2','Total_m2' ])
#print('test if worked',final_data['log_stellar_mass'][3])
final_data.to_csv(my_dir+'New_data_at_snap53')
d={'ID':all_ID,'log_stellar_mass':all_mass,'BT_ratio':all_circs,'half_stellar_radius':all_radius,'log_SFR':all_log_SFR,'log_sSFR':all_log_sSFR,'CAS_clump_idx':all_clump,'CAS_asym_idx':all_asym,'CAS_conc_idx':all_conc,'Blob_idx':all_blob,'Leaf_idx':all_leaf,'Med_clump_pos':med_rad_pos,'Clump_pos_IQR':IQR,'N_clumps':n_clumps,'Inner_m2':all_inner_struct,'Total_m2':all_total_struct}
dd.io.save(my_dir+'RS_data_at_snap53.h5',d)
'''                       


'''
#get  Bulge ratio and mass and log SFR and sSFR and stellar half mass radius for the descendants of SFGs at z=2 when they hit z=1. output into dataframe and csv
df=pd.read_csv(my_dir+'final_data_at_z2',header=0)
data=pd.DataFrame(df)
col_names=(data.columns)
print(col_names)
ID_at_z2=data['ID']
print('length of IDsis',len(ID_at_z2))
all_ID=([])
#all_clump=([])
#all_asym=([])
#all_conc=([])
#all_leaf=([])
#all_blob=([])
all_mass=([])
all_circs=([])
all_log_SFR=([])
all_log_sSFR=([])
all_radius=([])
#all_blob_pos=[]
no_circs=0
for i in range(len(ID_at_z2)):
    #get_descendants returns (ID,snap,lmstar,lrad,lSFR,lSSFR)
    descendants=get_descendants(33,ID_at_z2[i])
    #snapnum at z=1 is 50
    init_snapnum=50
    ID_at_z1=descendants[0][49]
    ID=ID_at_z1
    all_ID=np.append(all_ID,ID)
    print('doing this ID',ID)
    all_mass=np.append(all_mass,descendants[2][49])
    try:
        Final_BT=get_circs(ID,init_snapnum)
        all_circs=np.append(all_circs,Final_BT[1])
    except:
        print('no circs')
        Final_BT=0
        all_circs=np.append(all_circs,Final_BT)
        no_circs=+1
    all_log_SFR=np.append(all_log_SFR,descendants[4][49])
    all_log_sSFR=np.append(all_log_sSFR,descendants[5][49])
    all_radius=np.append(all_radius,descendants[3][49])
    #img=hdf.get_hdf(init_snapnum,ID,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    #img_array=hdf.get_lin_array(img)
    #lin_img=img_array[0]
    #log_img=img_array[1]
    #R_pet=hdf.get_petrosian(log_img,200,200)
    #d=hdf.get_dendro(lin_img,R_pet,200,200)
    #leaf_index=hdf.get_leaf_flux(lin_img,d)
    #blob_img=hdf.get_clumps_img(lin_img,R_pet,200,200)
    #blobs=hdf.get_clumps(blob_img)
    #blob_pos=[]
    #for pos in range(len(blobs[:,])):
    #    positionx=100-blobs[pos,1]
    #    positiony=100-blobs[pos,0]
    #    gal_r=np.sqrt(positionx**2+positiony**2)
    #    norm_gal_r=gal_r/R_pet
    #    blob_pos.append(norm_gal_r)
    #blob_mask=hdf.get_clump_mask(lin_img,blobs,200,200)
    #blob_index=hdf.get_clump_light(lin_img,blobs,blob_mask)
    #all_blob=np.append(all_blob,blob_index)
    #blob_pos=ID,blob_pos
    #blob_pos=tuple(blob_pos)
    #print(blob_pos)
    ##results in a seperate table with ID as odd idx and position radii as  even idx lists 
    #all_blob_pos=np.append(all_blob_pos,blob_pos)
    #print(all_blob_pos)
    #Clump_index=hdf.get_clump_index(lin_img,200,200,R_pet)
    #Asym_index=hdf.get_asymm(lin_img,200,200)
    #all_asym=np.append(all_asym,Asym_index)
    #Conc_index=hdf.get_CI(lin_img,200,200,R_pet)
    #all_conc=np.append(all_conc,Conc_index)
    #all_clump=np.append(all_clump,Clump_index)
    #all_leaf=np.append(all_leaf,leaf_index)
final_values=np.array([all_ID,all_mass,all_circs,all_radius,all_log_SFR,all_log_sSFR])
print('final value shape',final_values.shape)
#print('blob shape',all_blob_pos.shape)
final_values=np.transpose(final_values)
#output clump radial positions as a list
#with open(my_dir+'final_blob_pos_z1','w') as f:
#    for item in all_blob_pos:
#        f.write('%s \n'%item)
#all_blob_pos=np.transpose(all_blob_pos)
#print(all_blob_pos)
#final_pos_data=pd.DataFrame(all_blob_pos,columns=['ID','gal_cent_pos'])
final_data=pd.DataFrame(final_values,columns=['ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR'])
print('test if worked',final_data['log_stellar_mass'][3])
final_data.to_csv(my_dir+'final_data_at_z1')
'''

''' 
#For z=1 descendants - get the blob index values and CAS values and  dendrogram.
df=pd.read_csv(my_dir+'final_data_at_z1',header=0)
data=pd.DataFrame(df)
col_names=(data.columns)
print(col_names)
ID_at_z1=data['ID']
print(len(ID_at_z1))
all_clump=([])
all_asym=([])
all_conc=([])
all_leaf=([])
all_blob=([])
all_blob_pos=[]
for i in range(len(ID_at_z1)):
    print('there are this many left',len(ID_at_z1)-i)
    init_snapnum=50
    ID=int(ID_at_z1[i])
    print('doing this ID', ID)
    img=hdf.get_hdf(init_snapnum,ID,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    img_array=hdf.get_lin_array(img)
    lin_img=img_array[0]
    log_img=img_array[1]
    R_pet=hdf.get_petrosian(log_img,200,200)
    blob_img=hdf.get_clumps_img(lin_img,R_pet,200,200)
    blobs=hdf.get_clumps(blob_img)
    blob_pos=[]
    for pos in range(len(blobs[:,])):
        positionx=100-blobs[pos,1]
        positiony=100-blobs[pos,0]
        gal_r=np.sqrt(positionx**2+positiony**2)
        norm_gal_r=gal_r/R_pet
        blob_pos.append(norm_gal_r)
    blob_mask=hdf.get_clump_mask(lin_img,blobs,200,200)
    blob_index=hdf.get_clump_light(lin_img,blobs,blob_mask)
    all_blob=np.append(all_blob,blob_index)
    blob_pos=ID,blob_pos
    blob_pos=tuple(blob_pos)
    print(blob_pos)
    #results in a seperate table with ID as odd idx and position radii as  even idx lists 
    all_blob_pos=np.append(all_blob_pos,blob_pos)
    #print(all_blob_pos)
    d=hdf.get_dendro(lin_img,R_pet,200,200)
    leaf_index=hdf.get_leaf_flux(lin_img,d)
    Clump_index=hdf.get_clump_index(lin_img,200,200,R_pet)
    Asym_index=hdf.get_asymm(lin_img,200,200)
    all_asym=np.append(all_asym,Asym_index)
    Conc_index=hdf.get_CI(lin_img,200,200,R_pet)
    all_conc=np.append(all_conc,Conc_index)
    all_clump=np.append(all_clump,Clump_index)
    all_leaf=np.append(all_leaf,leaf_index)
final_values=np.array([all_clump,all_asym,all_conc,all_blob,all_leaf])
print(final_values.shape)
#print('blob shape',all_blob_pos.shape)
final_values=np.transpose(final_values)
#output clump radial positions as a list
with open(my_dir+'final_blob_pos_z1','w') as f:
    for item in all_blob_pos:
        f.write('%s \n'%item)
#all_blob_pos=np.transpose(all_blob_pos)
#print(all_blob_pos)
#final_pos_data=pd.DataFrame(all_blob_pos,columns=['ID','gal_cent_pos'])
final_data=pd.DataFrame(final_values,columns=['CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx'])
#print('test if worked',final_data['log_stellar_mass'][3])
final_data.to_csv(my_dir+'final_clump_data_at_z1')
'''

  
'''
#now read the blob pos file and plot against dataID of choice
#fig,ax=plt.subplots()
SFGz2=pd.read_csv(my_dir+'SFGs_10to115_at_z2_data',header=0)
SFGz1=pd.read_csv(my_dir+'SFGs_10to115_at_z1_data',header=0)
descendantsz1=pd.read_csv(my_dir+'descendants_at_z1_data',header=0)
col_names2=(SFGz2.columns)
col_names1=(SFGz1.columns)
print('z2',col_names2)
print('z1',col_names1)
del SFGz1['Unnamed: 0']
del SFGz2['Unnamed: 0']
del descendantsz1['Unnamed: 0']
BT_diff=descendantsz1['BT_ratio']-SFGz2['BT_ratio']
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,sharey=True,sharex=False)
with open(my_dir+'final_blob_pos_z2','r') as f:
    for (i,line) in enumerate(f):
        #print('line number',i)
        lines=line.rstrip()
        lines=lines.lstrip('[')
        lines=lines.rstrip(']')
        pos=np.array((lines.split(',')))
        #print(pos,pos.shape)
        if i%2==1:
            y=np.zeros(50)
            x1=np.linspace(0,1)
            #ax1.set_xlim(np.min(data_z1[col_names1[i]]),np.max(data_z1[col_names1[i]]))
#ax2.set_xlim(np.min(data_z2[col_names2[i]]),np.max(data_z2[col_names2[i]]))
            ax1.plot(x1,y,lw=0.5,color='k',ls='-')
            ID=(i-1)/2
            new_pos=np.array([])
            for i in range(len(pos)):
                pos_r=float(pos[i])
                new_pos=np.append(new_pos,pos_r)
            print(new_pos)
            med_pos=np.nanpercentile((new_pos),50)
            Q1_pos=np.nanpercentile(new_pos,25)
            Q3_pos=np.nanpercentile(new_pos,75)
            clump_n=len(pos)
            ax3.scatter(med_pos,BT_diff[ID],s=3,c='g')
            ax4.scatter(clump_n,BT_diff[ID],s=3,c='k')
            ax3.tick_params(labelbottom=True)
            ax3.get_xaxis().set_ticks([])
            ax4.tick_params(labelbottom=True)
            ax4.get_xaxis().set_ticks([])
            print('ID',ID)
            for r in range(len(pos)):
                pos_rad=float(pos[r])
                print(pos_rad,type(pos_rad))
                ax1.scatter(pos_rad,BT_diff[ID],s=3,c='b')
                ax1.tick_params(labelbottom=True)
                ax1.get_xaxis().set_ticks([])
with open(my_dir+'descendant_blob_pos_at_z1','r') as f:
    for (i,line) in enumerate(f):
        #print('line number',i)
        lines=line.rstrip()
        lines=lines.lstrip('[')
        lines=lines.rstrip(']')
        pos=np.array(lines.split(','))
        if len(pos)==0:
            pos=np.array(float(0))
#print(pos,pos.shape)
        if i%2==1:
            y=np.zeros(50)
            x2=np.linspace(0,1)
#ax1.set_xlim(np.min(data_z1[col_names1[i]]),np.max(data_z1[col_names1[i]]))
#ax2.set_xlim(np.min(data_z2[col_names2[i]]),np.max(data_z2[col_names2[i]]))
            ax2.plot(x2,y,lw=0.5,color='k',ls='-')
            ID=(i-1)/2
            print('ID',ID,'length',len(pos))
            new_pos=np.array([])
            for i in range(len(pos)):
                pos_r=float(pos[i])
                new_pos=np.append(new_pos,pos_r)
            med_pos=np.nanpercentile(new_pos,50)
            Q1_pos=np.nanpercentile(new_pos,25)
            Q3_pos=np.nanpercentile(new_pos,75)
            clump_n=len(pos)
            ax3.scatter(med_pos,BT_diff[ID],s=3,c='b')
            ax4.scatter(clump_n,BT_diff[ID],s=3,c='r')
            ax3.tick_params(labelbottom=True)
            ax3.get_xaxis().set_ticks([])
            ax4.tick_params(labelbottom=True)
            ax4.get_xaxis().set_ticks([])
            for r in range(len(pos)):
                try:
                    pos_rad=float(pos[r])
                except ValueError:
                    continue
                print(pos_rad,type(pos_rad))
                ax2.scatter(pos_rad,BT_diff[ID],s=3,c='r')
                ax2.tick_params(labelbottom=True)
                ax2.get_xaxis().set_ticks([])                
ax1.set_xlabel('Normalised Radial Position at z=2')
ax2.set_xlabel('Normalised Radial Position at z=1')
ax3.set_xlabel('Median Radial Position at z=1 and z=2')
ax4.set_xlabel('Number of clumps at z=1 and z=2')
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
ax1.set_ylabel('BT ratio difference z=1 - z=2')
plt.show()
fig.savefig(plot_dir+'Radial_position_vs_BT_ratio',dpi=300)
'''

'''                    
#now read the blob pos file and plot medians and num of clumps against dataID of choice. remember the blob files have alternate lines of ID followed by radial positions as strings. Need to convert.
#column hdrs are 'ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx' 
#fig,ax=plt.subplots()
SFGsnap26=pd.read_csv(my_dir+'New_data_at_snap26',header=0)
SFGsnap33=pd.read_csv(my_dir+'New_data_at_snap33',header=0)
SFGsnap40=pd.read_csv(my_dir+'New_data_at_snap40',header=0)
SFGsnap46=pd.read_csv(my_dir+'New_data_at_snap46',header=0)
#SFGz1=pd.read_csv(my_dir+'SFGs_10to115_at_z1_data',header=0)
#descendantsz1=pd.read_csv(my_dir+'descendants_at_z1_data',header=0)
col_names=(SFGsnap26.columns)
#BT_diff=descendantsz1['BT_ratio']-SFGz2['BT_ratio']
snaps=[26,33,40,46]
data=[SFGsnap26,SFGsnap33,SFGsnap40,SFGsnap46]
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,10),sharey=True,sharex=True,gridspec_kw={'wspace':0,'hspace':0,'top':0.95,'bottom':0.05,'left':0.05,'right':0.95})
for j,ax in enumerate(ax.ravel()):
    print(f'doing {j}')
    med_rad_pos=np.array([])
    Q1_rad_pos=np.array([])
    Q3_rad_pos=np.array([])
    n_clumps=np.array([])
    with open(my_dir+'New_blob_pos_snap'+str(snaps[j]),'r') as f:
        for (i,line) in enumerate(f):
            #print('line number',i)
            lines=line.rstrip()
            lines=lines.lstrip()
            lines=lines.lstrip('[')
            lines=lines.rstrip(']')
            lines=lines.strip(' ')
            pos=np.array((lines.split(',')))
            if len(pos)==0:
                pos=np.array(float(0))
        #print(pos,pos.shape)
            if i%2==1:
                new_pos=np.array([])
                for i in range(len(pos)):
                    pos_r=float(pos[i] or 0)
                    new_pos=np.append(new_pos,pos_r)
                #print(new_pos)
                med_pos=np.nanpercentile((new_pos),50)
                Q1_pos=np.nanpercentile(new_pos,25)
                Q3_pos=np.nanpercentile(new_pos,75)
                clump_n=len(pos)
                med_rad_pos=np.append(med_rad_pos,med_pos)
                Q1_rad_pos=np.append(Q1_rad_pos,Q1_pos)
                Q3_rad_pos=np.append(Q3_rad_pos,Q3_pos)
                n_clumps=np.append(n_clumps,clump_n)
            else:
                continue
    print(len(med_rad_pos))
    print(len(data[j]['Blob_idx']))
    IQR=Q3_rad_pos-Q1_rad_pos
    y=np.linspace(0,1,len(med_rad_pos))
    #ax1.scatter(BT_diff,med_rad_pos_z2,c='b',s=3,label='Median radial position at z=2')
    ax.errorbar(med_rad_pos,data[j]['Blob_idx'],xerr=IQR,fmt='o',ecolor='k',lw=0.15,ms=2,mfc='b',mec='b',label='Snap '+str(snaps[j]))
    x=np.full(len(med_rad_pos),0.4)
    ax.plot(x,y,ls='--')
#ax1.scatter(BT_diff,med_rad_pos_z1,c='r',s=3,label='Median radial positions of descendants at z=1')
    ax.set_xlabel('Median normalised radial position of clumps')
    ax.set_ylabel('Blob_idx')
    ax.legend()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
plt.show()
fig.savefig(plot_dir+'New_Median_radial_pos_vs_Blob')
'''

'''
#plot binned masses against n clumps
#column hdrs are 'ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx'
SFGz2=pd.read_csv(my_dir+'SFGs_10to115_at_z2_data',header=0)
descendantsz1=pd.read_csv(my_dir+'descendants_at_z1_data',header=0)
col_names=(SFGz2.columns)
del SFGz2['Unnamed: 0']
del descendantsz1['Unnamed: 0']
BT_diff=SFGz2['BT_ratio']-descendantsz1['BT_ratio']
all_clump_nums=([])
all_mass=np.array(SFGz2['log_stellar_mass'])
with open(my_dir+'blob_pos_SFGs_at_z2','r') as f:
    for (i,line) in enumerate(f):
        #print('line number',i)
        lines=line.rstrip()
        lines=lines.lstrip('[')
        lines=lines.rstrip(']')
        pos=np.array((lines.split(',')))
        #print(pos,pos.shape)
        if i%2==1:
            clump_num=len(pos)
        else:
            continue
        all_clump_nums=np.append(all_clump_nums,clump_num)
idx=np.argsort(BT_diff)
sorted_BT=BT_diff[idx]
sorted_mass=all_mass[idx]
sorted_clump_num=all_clump_nums[idx]
sorted_blob=np.log10(SFGz2['Blob_idx'][idx])
print(len(all_clump_nums))
fig,ax=plt.subplots()
len_list=0
mass_range=0
c_list=['r','b','k','g','c','y','m','darkcyan','olivedrab','salmon']
for diff in range(-5,5,1):
    mass_list=list(filter(lambda x: diff/10-0.1<=x<diff/10,sorted_BT))
    Gal_num=len(mass_list)
    print(Gal_num)
    clump_num_list=sorted_clump_num[len_list:len_list+Gal_num]
    #blob_list=sorted_blob[len_list:len_list+Gal_num]
    #len_list=len_list+Gal_num
    #print(len_list)
    #ax.scatter(blob_list,clump_num_list,s=4,c=c_list[mass_range],label='mass range'+str(mass_range))
    ax.hist(clump_num_list,10,color=c_list[mass_range],label='Log Mass'+str(-0.5+mass_range*0.1),histtype='bar',stacked=False,log=True)
    ax.legend()
    mass_range=mass_range+1
    print(mass_range)
ax.set_ylabel('Log N')
ax.set_xlabel('N-Clumps')
plt.show()    
fig.savefig(plot_dir+'histogram_BT_clump_num',dpi=300)            
'''


'''
#column hdrs are 'ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx'
#gas hdrslMgas -> log(Mgas) of all gas within 2 Rhalf_star_3D (directly from subhalo catalog)
# lMstar -> log(Mstar) of all stars within 2 Rhalf_star_3D (directly from subhalo catalog)
# fgas -> baryonic gas mass fraction computed from lMgas and lMstar
# centre -> 3-element array specifying centre used (by default subhaloPos, which is coordinate of particle with #most negative binding energy)
# Rhalf -> half-mass radius (by default Rhalf_star_3D directly from subhalo catalog)
# _lMstar -> log(Mstar) summing over all star particles that are being used in TNG_calc_inflow.py (currently wit#hin 5 Rhalf_star_3D)
# lMcoldgas -> log(Mgas) summing over all star-forming gas particles within (currently) 5 Rhalf_star_3D
# fcoldgas -> baryonic gas mass fraction computed from _lMstar and lMcoldgas
# gas_angmom3D -> angular momentum vector of gas particles (compute np.sum(gas_angmom3D**2, axis=1) to obtain th#e amplitude of the angular momentum vector)
# gas_specific_angmom3D -> specific angular momentum of gas (gas_angmom3D / 10**lMcoldgas)
# star_angmom3D -> angular momentum vector of star particles
# star_specific_angmom3D -> specific angular momentum of stars (star_angmom3D / 10**_lMstar)
# R_aperture -> array with aperture sizes considered for inflow metrics (units: kpc; use in combination with Rha#lf to determine which aperture corresponds to Re)
 #              default setting is for R_aperture entries for a given object to range from 0.1Re to 4Re in steps# of 0.1Re
# gas_radflow -> Msun/yr of gas flowing through an aperture (aperture definitions, see R_aperture)
# star_radflow -> Msun/yr of stars flowing through an aperture (aperture definitions, see R_aperture)
# gas_radvR -> radial velocity [km/s] of gas at a particular galactocentric radius
 #star_radvR -> radial velocity [km/s] of stars at a particular galactocentric radius
# frac_SFR1kpc -> fraction of total SFR within 1kpc radius
# frac_SFR -> fraction of total SFR within R_aperture radius
# gas_avg_vRv -> average radial-over-total velocity ratio for gas particles
# gas_avg_vzv -> average out-of-plane-over-total velocity ratio for gas particles
# gas_avg_vphiv -> average circular-over-total velocity ratio for gas particles
# star_avg_... -> equivalent for star particles
# gas_avg_v2Rv2 -> average energy-in-radial-over-total-energy ratio for gas particles

#more efficient way of cycling through plots
#SFGsnap26=pd.read_csv(my_dir+'New_data_at_snap26',header=0)
#SFGsnap33=pd.read_csv(my_dir+'New_data_at_snap33',header=0)
#SFGsnap40=pd.read_csv(my_dir+'New_data_at_snap40',header=0)
#SFGsnap46=pd.read_csv(my_dir+'New_data_at_snap46',header=0)
#SFGz1=pd.read_csv(my_dir+'SFGs_10to115_at_z1_data',header=0)
#descendantsz1=pd.read_csv(my_dir+'descendants_at_z1_data',header=0)
Snap26= pd.read_csv(my_dir+'New_data_at_snap26',header=0)
#for i in range(67):
#    del combined_data['all_BT'+str(i)]
#    del combined_data['lmbulge'+str(i)]
#del combined_data['subfindID']
#del combined_data['ID']
#del combined_data['Unnamed: 0']
#del combined_data['rel_BT_del_0.1']
#del combined_data['d_lmbulge_0.5']
#del combined_data['d_lmbulge_1.0']
#del combined_data['d_lmbulge_2.0']
#del combined_data['rel_BT_del_0.2']
#del combined_data['abs_BT_del_0.2']
#del combined_data['abs_BT_del_0.3']
#combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.1].index)
col_names=Snap26.columns
print(len(col_names))
#for col in col_names:
#    print(col)
#SFGz2['Blob_idx']=np.log10(SFGz2['Blob_idx'])
#BT_diff=descendantsz1['BT_ratio']-SFGz2['BT_ratio']#measure of growth over time
color_list=['blue','powderblue','lightblue','skyblue','lightskyblue','steelblue','dodgerblue','navy','darkblue','mediumblue']
fig,ax_list=plt.subplots(nrows=5,ncols=3,figsize=(10,6),sharey=True,sharex=False,gridspec_kw={'wspace':0,'hspace':0,'top':0.95,'bottom':0.05})
for i,ax in enumerate(ax_list.ravel()):
    print(i)
    sns.regplot(Snap26[col_names[i]],Snap26['Blob_idx'],color='lightgray',ax=ax,marker='o',scatter_kws={'s':3},line_kws={'lw':1})
    #sns.regplot(combined_data[col_names[i]],combined_data['d_BT_1.0'],color='dimgray',ax=ax,marker='o',scatter_kws={'s':3},line_kws={'lw':1},label='1.0 Gyrs')
    #sns.regplot(combined_data[col_names[i]],combined_data['d_BT_2.0'],color='black',ax=ax,marker='o',scatter_kws={'s':3},line_kws={'lw':1},label='2.0 Gyrs')
#    ax.scatter(SFGz2[hdr_list[i]],BT_diff,c='b',s=3,label=hdr_list[i])
    #ax.legend()
    ax.tick_params(top=False,bottom=False,left=False,right=False)
    ax.set_ylabel('Blob ')
    ax.set_xlabel(col_names[i])
    ax.get_xaxis().set_ticks([])
    ax.xaxis.set_label_coords(0.5,0.1)
#fig.text(0.1,0.5,'',va='center',rotation='vertical')
#fig,ax=plt.subplots()
#ax.scatter(np.log10(SFGz2['Blob_idx']),np.log10(SFGz2['CAS_clump_idx']),s=4,c='b',label='Blob vs CAS')
#ax.scatter(np.log10(SFGz2['Blob_idx']),np.log10(SFGz2['Leaf_idx']),s=4,c='r',label='Blob vs Leaf')
#ax.legend()
#ax.set_ylabel('Log index at z=2')
#ax.set_xlabel('Log index at z=2')
#plt.tight_layout(pad=0.2)
plt.legend(fontsize='x-small',labelspacing=0.2,columnspacing=1)
plt.show()    
#fig.savefig(plot_dir+'all_values_vs_BT_growth_lowmerger',dpi=600)               
'''
'''
#get the blob index and plot all original images ordered by clumpiness with Blob image and mock image and smoothed image and residual and identify clumps
#SFGsnap26=pd.read_csv(my_dir+'New_data_at_snap26',header=0)
#SFGsnap33=pd.read_csv(my_dir+'New_data_at_snap33',header=0)
#SFGsnap40=pd.read_csv(my_dir+'New_data_at_snap40',header=0)
#SFGsnap46=pd.read_csv(my_dir+'New_data_at_snap46',header=0)
combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.1].index)
combined_data=combined_data.drop(combined_data[combined_data['n_clumps']<3].index)
col_names=(combined_data.columns)
combined_data.sort_values('Blob_idx',inplace=True,ignore_index=True)
#print('z2',col_names2)
#SFGz2['Blob_idx']=np.log10(SFGz2['Blob_idx'])
nrows=5
ncols=5
fig,ax=plt.subplots(nrows,ncols,figsize=(10,10),sharey=False,sharex=False,gridspec_kw={'wspace':0,'hspace':0,'top':0.95,'bottom':0.05,'left':0.05,'right':0.95})
ax_title=['Original Image','Mock Image','Blurred Image','Residual Image','Clump Image']
ID_list=np.array(combined_data['ID'])
blob_list=np.array(combined_data['Blob_idx'])
#idx=np.argsort(blob_list)
#ID_list_sorted=ID_list[idx]
#blob_list_sorted=blob_list[idx]
rad_list=np.array(combined_data['half_stellar_radius'])
num=int(len(ID_list)/5)
#print(blob_list[0:8])
#print(ID_list[0:8])
init_snapnum=33
blob_values=[]
for i in range(nrows):
    print(f'i is {i}')
    ID_new=int(ID_list[i*num])
    blob_value=round(blob_list[i*num],4)
    blob_values.append(str(blob_value))
    print(ID_new,blob_value)
    #img=hdf.get_hdf(init_snapnum,ID_new,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    img=my_imgs+'grid_subhalo_TNG50-1_'+str(init_snapnum)+'_'+str(ID_new)+'.hdf5'
    img_array=hdf.get_lin_array(img)
    lin_img=img_array[0]
    log_img=2**(img_array[1]-35)
    #log_img=plt.imshow(log_img,cmap='viridis')
    R_pet=hdf.get_petrosian(log_img,200,200)
    print(f'R pet is {R_pet}')
    cent_1kpc=(40/rad_list[i])
    Re=40
    blur_rad=cent_1kpc*0.637
    blobs=hdf.get_clumps_img(lin_img,R_pet,200,200,blur_rad)
    Residual_img=blobs[1]
    Mock_img=2**(np.log10(blobs[2]))
    Gauss_image=blobs[3]
    clump_img=blobs[0]
    clumps=hdf.get_clumps(blobs[0])
    circle=plt.Circle((100,100),40,color='w',fill=False,ls='--')
    ax[i,0].imshow(log_img,cmap='viridis',aspect='auto')
    ax[i,0].set_xticks([])
    ax[i,0].set_yticks([])
    ax[i,0].add_patch(circle)
    ax[i,1].imshow(Mock_img,cmap='viridis',aspect='auto')
    ax[i,1].set_xticks([])
    ax[i,1].set_yticks([])
    circle=plt.Circle((100,100),40,color='w',fill=False,ls='--')
    ax[i,1].add_patch(circle)
    ax[i,2].imshow(Gauss_image,cmap='viridis',aspect='auto')
    ax[i,2].set_xticks([])
    ax[i,2].set_yticks([])
    circle=plt.Circle((100,100),40,color='w',fill=False,ls='--')
    ax[i,2].add_patch(circle)
    ax[i,3].imshow(Residual_img,cmap='viridis',aspect='auto')
    ax[i,3].set_xticks([])
    ax[i,3].set_yticks([])
    circle=plt.Circle((100,100),40,color='w',fill=False,ls='--')
    ax[i,3].add_patch(circle)
    ax[i,4].imshow(clump_img,cmap='viridis',aspect='auto')
    ax[i,4].set_xticks([])
    ax[i,4].set_yticks([])
    circle=plt.Circle((100,100),40,color='w',fill=False,ls='--')
    ax[i,4].add_patch(circle)
    for pos in range(len(clumps[:,])):
        position=((clumps[pos,1],clumps[pos,0]))
#multiply clump pixel radius by sqrt2 as returns fwhm.          
        radius=(clumps[pos,2]*np.sqrt(2))
        circle=plt.Circle(position,radius,color='w',fill=False)
        ax[i,3].add_patch(circle)
        circle=plt.Circle(position,radius,color='w',fill=False)
        ax[i,4].add_patch(circle)
        circle=plt.Circle(position,radius,color='w',fill=False)
        ax[i,0].add_patch(circle)
        circle=plt.Circle(position,radius,color='w',fill=False)
        ax[i,1].add_patch(circle)
#plt.tight_layout
    ax[i,0].text(15,15,str(blob_value),fontsize='small')
    ax[0,i].set_title(ax_title[i],y=1.05)
plt.subplots_adjust(hspace=0,wspace=0)
plt.show()        
fig.savefig(plot_dir+'new_blob_idx_processsnap33.png',dpi=1200,format='png')    
'''

'''
#get the blobiness in y axis and the redshift on x axis amd show image of blobbiest 
SFGsnap26=pd.read_csv(my_dir+'New_data_at_snap26',header=0)
SFGsnap33=pd.read_csv(my_dir+'New_data_at_snap33',header=0)
SFGsnap40=pd.read_csv(my_dir+'New_data_at_snap40',header=0)
SFGsnap46=pd.read_csv(my_dir+'New_data_at_snap46',header=0)
col_names2=(SFGsnap46.columns)
print('z2',col_names2)
#SFGz2['Blob_idx']=np.log10(SFGz2['Blob_idx'])
hdr_list=['ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx']
color_list=['blue','powderblue','lightblue','skyblue','lightskyblue','steelblue','dodgerblue','navy','darkblue','mediumblue']
nrows=5
ncols=4
fig,ax=plt.subplots(nrows,ncols,figsize=(8,10),sharey=False,sharex=False,gridspec_kw={'wspace':0,'hspace':0,'top':0.95,'bottom':0.05,'left':0.05,'right':0.95})
fig.suptitle('Snap 26                     Snap 33                      Snap 40                         Snap 46')
snaps=[26,33,40,46]
data=[SFGsnap26,SFGsnap33,SFGsnap40,SFGsnap46]
for j in range(ncols):
    idx=np.argsort(data[j]['Blob_idx'])
    ID_list=np.array(data[j]['ID'][idx])
    blob_list=np.array(data[j]['Blob_idx'][idx])
    rad_list=np.array(data[j]['half_stellar_radius'][idx])
#print(blob_list[0:8])
#print(ID_list[0:8])
    init_snapnum=snaps[j]
    blob_values=[]
    for i in range(nrows):
        print(f'i is {i}')
        ID_new=int(ID_list[-1-i])
        blob_value=round(blob_list[-1-i],3)
        blob_values.append(str(blob_value))
        print(ID_new,blob_value)
        img=my_imgs+'grid_subhalo_TNG50-1_'+str(init_snapnum)+'_'+str(ID_new)+'.hdf5'
        img_array=hdf.get_lin_array(img)
        lin_img=img_array[0]
        log_img=2**(img_array[1]-35)
        R_pet=200
        cent_1kpc=(40/rad_list[i])
        blur_rad=cent_1kpc*0.637
        blobs=hdf.get_clumps_img(lin_img,R_pet,200,200,blur_rad)
        smooth_img=blobs[1]
        blurred_img=blobs[2]
        clump_img=blobs[0]
        clumps=hdf.get_clumps(blobs[0])
        ax[i,j].imshow(log_img,cmap='viridis',aspect='auto')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        for pos in range(len(clumps[:,])):
            position=((clumps[pos,1],clumps[pos,0]))
#multiply clump pixel radius by sqrt2 as returns fwhm.          
            radius=(clumps[pos,2]*np.sqrt(2))
            circle=plt.Circle(position,radius,color='w',fill=False)
            ax[i,j].add_patch(circle)
#plt.tight_layout
        ax[i,j].set_title(str(blob_value),pad=-0.5,y=0.9)
plt.subplots_adjust(hspace=0,wspace=0)
plt.show()
'''
'''
#plot all original images ordered by clumpiness with the descendant image and clumps. Accidentally edited this to do blurred inage and residual but principle is the same. 
SFGz2=pd.read_csv(my_dir+'New_data_at_z2',header=0)
#descendantsz1=pd.read_csv(my_dir+'descendants_at_z1_data',header=0)
col_names2=(SFGz2.columns)
print('z2',col_names2)
#SFGz2['Blob_idx']=np.log10(SFGz2['Blob_idx'])
hdr_list=['ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx']
color_list=['blue','powderblue','lightblue','skyblue','lightskyblue','steelblue','dodgerblue','navy','darkblue','mediumblue']
nrows=5
ncols=3
fig,ax=plt.subplots(nrows,ncols,figsize=(6,10),sharey=False,sharex=False,gridspec_kw={'wspace':0,'hspace':0,'top':0.95,'bottom':0.05,'left':0.05,'right':0.95})
fig.suptitle('Galaxy at z=2    Residual      Clumps')
idx=np.argsort(SFGz2['Blob_idx'])
ID_list=np.array(SFGz2['ID'][idx])
blob_list=np.array(SFGz2['Blob_idx'][idx])
rad_list=np.array(SFGz2['half_stellar_radius'][idx])
#ID_list_descendant=np.array(descendantsz1['ID'][idx])
#print(blob_list[0:8])
#print(ID_list[0:8])
init_snapnum=33
blob_values=[]
for i in range(nrows):
    print(f'i is {i}')
    ID_new=int(ID_list[i*65])
    blob_value=round(blob_list[i*65],3)
    #ID_descendant=int(ID_list_descendant[i*60])
    print(ID_new,blob_value)
    #img=hdf.get_hdf(init_snapnum,ID_new,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    img=my_imgs+'grid_subhalo_TNG50-1_33_'+str(ID_new)+'.hdf5'
    #desc_img=my_imgs+'grid_subhalo_TNG50-1_50_'+str(ID_descendant)+'.hdf5'
    img_array=hdf.get_lin_array(img)
    #desc_img_array=hdf.get_lin_array(desc_img)
    lin_img=img_array[0]
    log_img=2**(img_array[1]-35)
    #desc_lin_img=desc_img_array[0]
    #desc_log_img=2**(desc_img_array[1]-35)
    #log_img=plt.imshow(log_img,cmap='viridis')
    R_pet=200
    cent_1kpc=(40/rad_list[i])
    blur_rad=cent_1kpc*0.637
    blobs=hdf.get_clumps_img(lin_img,R_pet,200,200,blur_rad)
    smooth_img=blobs[1]
    blurred_img=blobs[2]
    #desc_blobs=hdf.get_clumps_img(desc_lin_img,R_pet,200,200)
    #desc_smooth_img=desc_blobs[1]
    clumps=hdf.get_clumps(blobs[0])
    #desc_clumps=hdf.get_clumps(desc_blobs[0])
    #print('length of clumps',len(desc_clumps[:,]),desc_clumps)
    #log_img=plt.imshow(log_img,cmap='viridis')
    ax[i,0].imshow(blurred_img,cmap='viridis',aspect='auto')
    ax[i,0].set_xticks([])
    ax[i,0].set_yticks([])
    ax[i,1].imshow(smooth_img,cmap='viridis',aspect='auto')
    ax[i,1].set_xticks([])
    ax[i,1].set_yticks([])
    ax[i,2].imshow(blobs[0],cmap='viridis',aspect='auto')
    ax[i,2].set_xticks([])
    ax[i,2].set_yticks([])
    #ax[i,3].imshow(desc_smooth_img,cmap='viridis',aspect='auto')
    #ax[i,3].set_xticks([])
    #ax[i,3].set_yticks([])
    for pos in range(len(clumps[:,])):
        position=((clumps[pos,1],clumps[pos,0]))
#multiply clump pixel radius by sqrt2 as returns fwhm.          
        radius=(clumps[pos,2]*np.sqrt(2))
        circle=plt.Circle(position,radius,color='w',fill=False)
        ax[i,0].add_patch(circle)
    #for pos in range(len(desc_clumps[:,])):
    #    position=((desc_clumps[pos,1],desc_clumps[pos,0]))
#multiply clump pixel radius by sqrt2 as returns fwhm.          
    #    radius=(desc_clumps[pos,2]*np.sqrt(2))
    #    circle=plt.Circle(position,radius,color='w',fill=False)
    #    ax[i,2].add_patch(circle)    
#plt.tight_layout
    ax[i,0].set_title(str(blob_value),pad=-0.5,y=0.9)
plt.subplots_adjust(hspace=0,wspace=0)
plt.show()        
fig.savefig(plot_dir+'new_blob_visuals',dpi=600)
'''
'''
#This code takes Stijn's data nd combines it with my data into pandas files. Have to selct correct Ids and put into reshaped numpys first.Then turn in to datframe and concaneta. Can then use random forest.
#gas headers are ['R_aperture', 'Rhalf', '_lMstar', 'centre', 'dir_plot', 'fcoldgas', 'fgas', 'frac_SFR', 'frac_SFR1kpc', 'gas_angmom3D', 'gas_avg_v2Rv2', 'gas_avg_v2phiv2', 'gas_avg_v2zv2', 'gas_avg_vRv', 'gas_avg_vphiv', 'gas_avg_vzv', 'gas_radflow', 'gas_radvR', 'gas_specific_angmom3D', 'lMcoldgas', 'lMgas', 'lMstar', 'redshift', 'sim_name', 'snapNum', 'star_angmom3D', 'star_avg_v2Rv2', 'star_avg_v2phiv2', 'star_avg_v2zv2', 'star_avg_vRv', 'star_avg_vphiv', 'star_avg_vzv', 'star_radflow', 'star_radvR', 'star_specific_angmom3D', 'subfindID']
#Bt headers are ['BT', 'all_BT', 'all_ages', 'all_lMbulge', 'delBT', 'delBT_thresh', 'dellMbulge', 'dellMbulge_thresh', 'delt_thresh', 'lMstar', 'lSFR', 'tbulgegrowth_abs', 'tbulgegrowth_rel']
SFGsnap26=pd.read_csv(my_dir+'New_data_at_snap26',header=0)
SFGsnap33=pd.read_csv(my_dir+'New_data_at_snap33',header=0)
SFGsnap40=pd.read_csv(my_dir+'New_data_at_snap40',header=0)
SFGsnap46=pd.read_csv(my_dir+'New_data_at_snap46',header=0)
hdr_list=['ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx']
gas_file=my_dir+'gas_Mgt9.0_SFG_BTlt03_sn33.h5'
gas_list=my_dir+'Mgt9.0_SFG_BTlt03_sn33.lis'
Bt_file=my_dir+'bulgeprop_Mgt9.0_All_sn33.h5'
Bt_list=my_dir+'Mgt9.0_All_sn33.lis'
BT_data=hdf.read_hdf(Bt_file,0)
all_ages=hdf.read_hdf(Bt_file,2)[1]
del_thresholds=hdf.read_hdf(Bt_file,8)[1]#these are  3 time intervals(0.5,1.2)v change in bt
del_BT_hdrs=[]
for thresh in del_thresholds:
    del_BT_hdrs.append('d_BT_'+str(thresh))
del_lmbulge_hdrs=[]
for thresh in del_thresholds:
    del_lmbulge_hdrs.append('d_lmbulge_'+str(thresh))    
print('time thresh',del_thresholds)
rel_BT_thresh=hdf.read_hdf(Bt_file,5)[1]#these are 2 Bt ratio deltas 0.1 and 0.2. Value is time to reach
rel_BT_thresh_hdrs=[]
for thresh in rel_BT_thresh:
    rel_BT_thresh_hdrs.append('rel_BT_del_'+str(thresh))
print('BT_rel thresh',rel_BT_thresh)
abs_BT_thresh=hdf.read_hdf(Bt_file,7)[1]#These are 2 log mass thresholds 0.2 and 0.3 value is time to reach
abs_BT_thresh_hdrs=[]
for thresh in abs_BT_thresh:
    abs_BT_thresh_hdrs.append('abs_BT_del_'+str(thresh))
print('abs_BT_thresh',abs_BT_thresh)
BT_hdrs=['all_BT', 'all_lMbulge', 'delBT', 'dellMbulge','tbulgegrowth_abs', 'tbulgegrowth_rel']
all_BT_hdrs=BT_data[0]
BT_num=[]
gas_num=[]
for i,name in enumerate(all_BT_hdrs):
    for j in range(len(BT_hdrs)):
        if all_BT_hdrs[i]==BT_hdrs[j]:
            BT_num.append(i)
#print(BT_num)
gas_data=hdf.read_hdf(gas_file,0)
gas_IDs=hdf.read_lis(gas_list)[1]['col2']
#print('gas_ID_len',len(gas_IDs))
bt_IDs=hdf.read_lis(Bt_list)[1]['col2']
gas_hdrs=['fcoldgas', 'fgas', 'frac_SFR1kpc', 'gas_avg_v2Rv2', 'gas_avg_v2phiv2', 'gas_avg_v2zv2', 'gas_avg_vRv', 'gas_avg_vphiv', 'gas_avg_vzv', 'lMcoldgas', 'lMgas',  'star_avg_v2Rv2', 'star_avg_v2phiv2', 'star_avg_v2zv2', 'star_avg_vRv', 'star_avg_vphiv', 'star_avg_vzv', 'subfindID']
all_gas_hdrs=gas_data[0]
for i,name in enumerate(all_gas_hdrs):
    for j in range(len(gas_hdrs)):
        if all_gas_hdrs[i]==gas_hdrs[j]:
            gas_num.append(i)
#print(gas_num)
BT_data_snaps=np.array([])
all_lmbulges=np.array([])
all_delBT_vals=np.array([])
all_delMbulge_vals=np.array([])
all_tbulgegrowth_abs_vals=np.array([])
all_tbulgegrowth_rel_vals=np.array([])
all_gas_vals=np.array([])
t=0
for i in range(len(SFGsnap33['ID'])):
    all_k=[]
    #print(SFGsnap33['ID'][i])
    for j in range(len(bt_IDs)):
        if int(SFGsnap33['ID'][i])==int(bt_IDs[j]):
            BT_vals=hdf.read_hdf(Bt_file,BT_num[0])
            all_BT_vals=BT_vals[1][j]
            BT_data_snaps=np.append(BT_data_snaps,all_BT_vals)

            BT_vals=hdf.read_hdf(Bt_file,BT_num[1])
            lmbulges=BT_vals[1][j]
            all_lmbulges=np.append(all_lmbulges,lmbulges)

            BT_vals=hdf.read_hdf(Bt_file,BT_num[2])
            delBT_vals=BT_vals[1][j]
            all_delBT_vals=np.append(all_delBT_vals,delBT_vals)

            BT_vals=hdf.read_hdf(Bt_file,BT_num[3])
            dellMbulge=BT_vals[1][j]
            all_delMbulge_vals=np.append(all_delMbulge_vals,dellMbulge)
            
            BT_vals=hdf.read_hdf(Bt_file,BT_num[4])
            bulgegrowth_abs=BT_vals[1][j]
            all_tbulgegrowth_abs_vals=np.append(all_tbulgegrowth_abs_vals,bulgegrowth_abs)
            
            BT_vals=hdf.read_hdf(Bt_file,BT_num[5])
            bulgegrowth_rel=BT_vals[1][j]
            all_tbulgegrowth_rel_vals=np.append(all_tbulgegrowth_rel_vals,bulgegrowth_rel)
        else:
            continue
    for k in range(len(gas_IDs)):
        if int(SFGsnap33['ID'][i])==int(gas_IDs[k]):
            #print('number that fit----------------------------------i is ',i,SFGsnap33['ID'][i],gas_IDs[k])
            for l in range(len(gas_num)):
                gas_vals=hdf.read_hdf(gas_file,gas_num[l])
                val=gas_vals[1][k]
                #print(gas_vals[0][gas_num[l]],val)
                all_gas_vals=np.append(all_gas_vals,val)
        else:
            all_k.append(k)
            continue
    #print('gas ID looked at length is ',len(all_k))
    if len(all_k)==454:
        t+=1
        all_gas_vals=np.append(all_gas_vals,np.full(18,pd.NA))
    #print('t is num with no match', t)    
BT_data_snaps=BT_data_snaps.reshape(272,67)
print('numpy array',BT_data_snaps.shape)
all_lmbulges=all_lmbulges.reshape(272,67)
all_delBT_vals=all_delBT_vals.reshape(272,3)
all_delMbulge_vals=all_delMbulge_vals.reshape(272,3)
all_tbulgegrowth_abs_vals=all_tbulgegrowth_abs_vals.reshape(272,2)
all_tbulgegrowth_rel_vals=all_tbulgegrowth_rel_vals.reshape(272,2)
#print('length is ',t)
all_gas_vals=all_gas_vals.reshape(272,len(gas_hdrs))
print(all_gas_vals.shape)
Bt_snap_hdrs=[]
lmbulge_hdrs=[]
for i in range(67):
    BT_snap_hdr='all_BT'+str(i)
    lmbulge_hdr='lmbulge'+str(i)
    Bt_snap_hdrs.append(BT_snap_hdr)
    lmbulge_hdrs.append(lmbulge_hdr)
BT_snap_data=pd.DataFrame(BT_data_snaps,columns=Bt_snap_hdrs)
print(BT_snap_data.shape)
all_lmbulges_data=pd.DataFrame(all_lmbulges,columns=lmbulge_hdrs)
all_delBT_data=pd.DataFrame(all_delBT_vals,columns=del_BT_hdrs)
all_delMbulge_data=pd.DataFrame(all_delMbulge_vals,columns=del_lmbulge_hdrs)
all_tbulgegrowth_abs_data=pd.DataFrame(all_tbulgegrowth_abs_vals,columns=abs_BT_thresh_hdrs)
all_tbulgegrowth_rel_data=pd.DataFrame(all_tbulgegrowth_rel_vals,columns=rel_BT_thresh_hdrs)
all_gas_data=pd.DataFrame(all_gas_vals,columns=gas_hdrs)
#print('dataframe',all_gas_data)
combined_data=pd.concat([BT_snap_data,all_lmbulges_data,all_delBT_data,all_delMbulge_data,all_tbulgegrowth_abs_data,all_tbulgegrowth_rel_data,all_gas_data,SFGsnap33],axis=1,ignore_index=False)
for hdr in combined_data.columns:
    print(hdr)
medians=combined_data.median()
#for i,col in enumerate(combined_data.columns):
#    print(col)
#    combined_data[col].fillna(medians[i],inplace=True)
#print(combined_data)
#print(nan_removed['subfindID'])
#print(nan_removed)
combined_data.to_csv(my_dir+'Combined_data_snap33')

'''

#This code takes Stijn's updated parameter  data and combines it with my data into pandas files. I have updated the read hdf to use deepdish so the return is the dictionary object of alldata. access keys through keys() attribute . Need to use gas_prop IDs as  the basic source of IDs. read_lis returns list of columns in first param and then data list in second return. combine all 1-d data and BTdel over 0.5,1,2,4 Gyr  into new h5 file for all snaps.   Can then use random forest.
#gas headers are ['R_aperture', 'Rhalf','SWbar','SWrbar', '_lMstar', 'centre', 'dir_plot','dynbar','fcoldM_inRe', 'fcoldgas','fcoldgas_inRe', 'fgas', 'frac_SFR', 'frac_SFR1kpc','frac_coldgas','frac_coldgas1kpc', 'gas_angmom3D', 'gas_avg_v2Rv2', 'gas_avg_v2phiv2', 'gas_avg_v2zv2', 'gas_avg_vRv', 'gas_avg_vphiv', 'gas_avg_vzv', 'gas_radflow', 'gas_radvR', 'gas_specific_angmom3D', 'lMcoldgas', 'lMgas', 'lMstar','lSFR','logt_gas_inflow','logt_star_inflow','new_lMgas_cold','new_lMgas_tot', 'redshift', 'sim_name', 'snapNum','snap_fut', 'star_angmom3D', 'star_avgRe_v2Rv2', 'star_avgRe_v2phiv2', 'star_avgRe_v2zv2', 'star_avgRe_vRv', 'star_avgRe_vphiv', 'star_avgRe_vzv','star_avg_v2Rv2', 'star_avg_v2phiv2', 'star_avg_v2zv2', 'star_avg_vRv', 'star_avg_vphiv', 'star_avg_vzv','star_distR', 'star_radflow', 'star_radvR', 'star_specific_angmom3D', 'subfindID'] 
#Bt headers are ['BT', 'all_BT', 'all_ages', 'all_lMbulge', 'delBT', 'delBT_thresh', 'dellMbulge', 'dellMbulge_thresh', 'delt_thresh', 'lMstar', 'lSFR', 'tbulgegrowth_abs', 'tbulgegrowth_rel']
snap_list=[26,29,33,36,40,46,53]
BT_time_list=[0.5,1,2,4]
for snap in snap_list:
    SFGsnap_data=hdf.read_hdf(my_dir+'RS_data_at_snap'+str(snap)+'.h5')
    #print(SFGsnap_data)
    Gas_snap_data=hdf.read_hdf(output_dir+'SFGMgt10_BTlt03_noM/gasprop/gas_Mgt10.0_SFG_BTlt03_noM_sn'+str(snap)+'.h5')
    BT_snap_data=hdf.read_hdf(output_dir+'bulgeprop/bulgeprop_Mgt9.0_All_sn'+str(snap)+'.h5')
    BT_IDsnap=hdf.read_lis(list_dir+'subfindID/Mgt9.0_All_sn'+str(snap)+'.lis')[1]['col2']
    merger_snap_data=hdf.read_hdf(output_dir+'mergers/merger_Mgt9.0_All_sn'+str(snap)+'.h5')
    #print(merger_snap_data['delt_thresh'])
    #just use summed mass ratio from merger data.can use the other keys also. there is only one delt thresh at 2gyr
    merger_snap_data=np.ravel(merger_snap_data['sum_mass_ratio'])
    #print(len(merger_snap_data))
    BTidx=[]
    SFGidx=[]
    for i in range(len(BT_IDsnap)):
        for j in range(len(Gas_snap_data['subfindID'])):
            if int(BT_IDsnap[i])==int(Gas_snap_data['subfindID'][j]):
                BTidx.append(i)
            else:
                continue
    for k in range(len(SFGsnap_data['ID'])):
        for l in range(len(Gas_snap_data['subfindID'])):
            if int(SFGsnap_data['ID'][k])==int(Gas_snap_data['subfindID'][l]):
                SFGidx.append(k)
            else:
                continue
    #print('BT idx len',len(BTidx))
    #print('SFG idx len',len(SFGidx))
#print('bt vals', BT_snap33_data['delBT'][BTidx])
    combined_d={}
    for key in Gas_snap_data.keys():
        if len(Gas_snap_data[key])==len(Gas_snap_data['subfindID']):
            combined_d.update({key:Gas_snap_data[key]})
            #print(key,Gas_snap_data[key])
        else:
            continue
#print('updated keys',combined_d.keys())
    combined_d.update({'delBT':BT_snap_data['delBT'][BTidx]})
    combined_d.update({'summed_mass_ratio':merger_snap_data[BTidx]})
    for key in SFGsnap_data.keys():
        combined_d.update({key:SFGsnap_data[key][SFGidx]})
    for key in combined_d.keys():
        print(snap,key,combined_d[key].shape)
    dd.io.save(my_dir+'combined_data_at_snap'+str(snap)+'.h5',combined_d)               
    
        



'''
#get the structure plot via hdf.get_structure and plot beside Halpha image 
SFGsnap26=pd.read_csv(my_dir+'New_data_at_snap26',header=0)
SFGsnap33=pd.read_csv(my_dir+'New_data_at_snap33',header=0)
SFGsnap40=pd.read_csv(my_dir+'New_data_at_snap40',header=0)
SFGsnap46=pd.read_csv(my_dir+'New_data_at_snap46',header=0)
col_names2=(SFGsnap46.columns)
#SFGz2['Blob_idx']=np.log10(SFGz2['Blob_idx'])
hdr_list=['ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx']
nrows=5
ncols=3
fig,ax=plt.subplots(nrows,ncols,figsize=(6,10),sharey=False,sharex=False,gridspec_kw={'wspace':0,'hspace':0,'top':0.95,'bottom':0.05,'left':0.05,'right':0.95})
fig.suptitle('Log Original Image      Structure Plot      Clumps' )
idx=np.argsort(SFGsnap33['Blob_idx'])
ID_list=np.array(SFGsnap33['ID'][idx])
blob_list=np.array(SFGsnap33['Blob_idx'][idx])
rad_list=np.array(SFGsnap33['half_stellar_radius'][idx])
num=int(len(ID_list)/nrows)-2
#print(blob_list[0:8])
#print(ID_list[0:8])
init_snapnum=33
blob_values=[]
for i in range(nrows):
    print(f'i is {i}')
    ID_new=int(ID_list[i*num])
    blob_value=round(blob_list[i*num],5)
    blob_values.append(str(blob_value))
    print(ID_new,blob_value)
    img=my_imgs+'grid_subhalo_TNG50-1_'+str(init_snapnum)+'_'+str(ID_new)+'.hdf5'
    img_array=hdf.get_lin_array(img)
    lin_img=img_array[0]
    log_img=2**(img_array[1]-35)
    #log_img=plt.imshow(log_img,cmap='viridis')
    struct=hdf.get_structure(lin_img,r=100,dr=2,theta=360)
    circle=plt.Circle((100,100),40,color='red',fill=False,ls='--')
    ax[i,1].plot(struct[1],struct[2],color='blue',lw=0.5,ls='-')
    ax[i,1].set_xlabel('Radius(pixels)')
    ax[i,1].set_yticks([])
    x=np.full(len(struct[0]),40)
    y=np.linspace(np.min(struct[2]),np.max(struct[2]),len(struct[2]))
    ax[i,1].plot(x,y,color='red',lw=0.5,ls='--')
    ax[i,1].set_ylim(np.min(struct[2]),np.max(struct[2]))
    R_pet=200
    cent_1kpc=(40/rad_list[i])
    blur_rad=cent_1kpc*0.637
    blobs=hdf.get_clumps_img(lin_img,R_pet,200,200,blur_rad)
    smooth_img=blobs[1]
    blurred_img=blobs[2]
    #desc_blobs=hdf.get_clumps_img(desc_lin_img,R_pet,200,200)
    #desc_smooth_img=desc_blobs[1]
    clumps=hdf.get_clumps(blobs[0])
    #desc_clumps=hdf.get_clumps(desc_blobs[0])
    #print('length of clumps',len(desc_clumps[:,]),desc_clumps)
    #log_img=plt.imshow(log_img,cmap='viridis')
    ax[i,0].imshow(log_img,cmap='viridis',aspect='auto')
    ax[i,0].add_patch(circle)
    ax[i,0].set_xticks([])
    ax[i,0].set_yticks([])
    circle=plt.Circle((100,100),40,color='red',fill=False,ls='--')
    ax[i,2].imshow(blurred_img,cmap='viridis',aspect='auto')
    ax[i,2].add_patch(circle)
    ax[i,2].set_xticks([])
    ax[i,2].set_yticks([])
    #ax[i,3].imshow(desc_smooth_img,cmap='viridis',aspect='auto')
    #ax[i,3].set_xticks([])
    #ax[i,3].set_yticks([])
    for pos in range(len(clumps[:,])):
        position=((clumps[pos,1],clumps[pos,0]))
#multiply clump pixel radius by sqrt2 as returns fwhm.          
        radius=(clumps[pos,2]*np.sqrt(2))
        circle=plt.Circle(position,radius,color='w',fill=False)
        ax[i,2].add_patch(circle)
    #for pos in range(len(desc_clumps[:,])):
    #    position=((desc_clumps[pos,1],desc_clumps[pos,0]))
#multiply clump pixel radius by sqrt2 as returns fwhm.          
    #    radius=(desc_clumps[pos,2]*np.sqrt(2))
    #    circle=plt.Circle(position,radius,color='w',fill=False)
    #    ax[i,2].add_patch(circle)    
#plt.tight_layout
    ax[i,0].set_title(str(blob_value),pad=-0.5,y=0.9)
                     
plt.subplots_adjust(hspace=0,wspace=0)
plt.show()        
fig.savefig(plot_dir+'structure_testsnap33',dpi=600)    
'''

'''
#get inner struct and compare. m=2
SFGsnap33=pd.read_csv(my_dir+'Combined_data_snap33',header=0)
col_names=SFGsnap33.columns
print(col_names)
del SFGsnap33['Unnamed: 0']
ID_list=np.array(SFGsnap33['ID'])
init_snapnum=33
#for name in col_names:
#    print(name)
all_inner_struct=np.array([])
all_total_struct=np.array([])
for i in range(len(ID_list)):
    print(f'---------------------------------------------------i is {i}. There are {len(ID_list)-i} left')
    ID_new=int(ID_list[i])
    img=my_imgs+'grid_subhalo_TNG50-1_'+str(init_snapnum)+'_'+str(ID_new)+'.hdf5'
    img_array=hdf.get_lin_array(img)
    lin_img=img_array[0]
    log_img=2**(img_array[1]-35)
    #log_img=plt.imshow(log_img,cmap='viridis')
    struct=hdf.get_structure(lin_img,r=100,dr=2,theta=180)
    total_struct=np.sum(struct[2])
    inner_struct=np.sum(struct[2][0:20])/np.sum(struct[2])
    all_inner_struct=np.append(all_inner_struct,inner_struct)
    all_total_struct=np.append(all_total_struct,total_struct)
all_struct=np.array((all_inner_struct,all_total_struct))
all_struct=np.transpose(all_struct)
print(all_struct.shape)
all_inner=pd.DataFrame(all_struct,columns=['inner_struct_frac','total_struct'])
all_inner.to_csv(my_dir+'inner_struct_data_snap33')
print('test if worked',all_inner['inner_struct_frac'][3])
'''                       
'''
#get only galaxies with small future mergers and pput into csv.
SFGsnap33=pd.read_csv(my_dir+'Combined_data_snap33',header=0)
col_names=SFGsnap33.columns
print(col_names)
del SFGsnap33['Unnamed: 0']
combined_ID_list=np.array(SFGsnap33['ID'])
#print(combined_ID_list)
summed_merger=hdf.read_hdf(my_dir+'merger_Mgt9.0_All_sn33.h5',2)
all_summed_merger=summed_merger[1]
#print(all_summed_merger.shape)
idx=np.where(all_summed_merger<0.01)[0]
#print(idx.shape)
IDlist=hdf.read_lis(my_dir+'Mgt9.0_All_sn33.lis')
mergerlist=IDlist[1]['col2']
#print(mergerlist.size)
low_merger_list=mergerlist[idx]
all_mergers=np.array([])
#print(low_merger_list.size)
for i in range(len(combined_ID_list)):
    print(i)
    for j in range(len(mergerlist)):
        if int(combined_ID_list[i])==int(mergerlist[j]):
            print(f'GALAXY fOUND for {i} merger is {all_summed_merger[:,j]}')
            all_mergers=np.append(all_mergers,all_summed_merger[:,j])
        else:
            continue
print(all_mergers.size,all_mergers)
mergers=pd.DataFrame(all_mergers,columns=['summed_merger_ratio'])
mergers.to_csv(my_dir+'merger_hx')
'''
'''
#get the medial positions as an np array and save as csv
SFGsnap26=pd.read_csv(my_dir+'New_data_at_snap26',header=0)
SFGsnap33=pd.read_csv(my_dir+'New_data_at_snap33',header=0)
SFGsnap40=pd.read_csv(my_dir+'New_data_at_snap40',header=0)
SFGsnap46=pd.read_csv(my_dir+'New_data_at_snap46',header=0)
combined_data=pd.read_csv(my_dir+'Combined_data_snap33',header=0)
#print(combined_data['ID'],SFGsnap33['ID'])

#SFGz1=pd.read_csv(my_dir+'SFGs_10to115_at_z1_data',header=0)
#descendantsz1=pd.read_csv(my_dir+'descendants_at_z1_data',header=0)
col_names=(SFGsnap26.columns)
#BT_diff=descendantsz1['BT_ratio']-SFGz2['BT_ratio']
snaps=33
data=SFGsnap33
med_rad_pos=np.array([])
Q1_rad_pos=np.array([])
Q3_rad_pos=np.array([])
all_IQR=np.array([])
n_clumps=np.array([])
with open(my_dir+'New_blob_pos_snap'+str(snaps),'r') as f:
    for (i,line) in enumerate(f):
        #print('line number',i)
        lines=line.rstrip()
        lines=lines.lstrip()
        lines=lines.lstrip('[')
        lines=lines.rstrip(']')
        lines=lines.strip(' ')
        pos=np.array((lines.split(',')))
        if len(pos)==0:
            pos=np.array(float(0))
    #print(pos,pos.shape)
        if i%2==1:
            new_pos=np.array([])
            for i in range(len(pos)):
                pos_r=float(pos[i] or 0)
                new_pos=np.append(new_pos,pos_r)
            #print(new_pos)
            med_pos=np.nanpercentile((new_pos),50)
            Q1_pos=np.nanpercentile(new_pos,25)
            Q3_pos=np.nanpercentile(new_pos,75)
            IQR=Q3_pos-Q1_pos
            clump_n=len(pos)
            all_IQR=np.append(all_IQR,IQR) 
            med_rad_pos=np.append(med_rad_pos,med_pos)
            n_clumps=np.append(n_clumps,clump_n)
        else:
            continue
all_pos=np.array((n_clumps,med_rad_pos,all_IQR))
all_pos=np.transpose(all_pos)
print(all_pos.shape)
blob_possnap33=pd.DataFrame(all_pos,columns=['n_clumps','med_pos','IQR'])
print(blob_possnap33['med_pos'])
blob_possnap33.to_csv(my_dir+'blob_possnap33')
                           
#combined combined data with blob pos and merger hx and struct
combined_data=pd.read_csv(my_dir+'Combined_data_snap33',header=0)
blob_pos=pd.read_csv(my_dir+'blob_possnap33',header=0)
struct_data=pd.read_csv(my_dir+'inner_struct_data_snap33',header=0)
merger_data=pd.read_csv(my_dir+'merger_hx',header=0)
all_combined_data=pd.concat([combined_data,blob_pos,struct_data,merger_data],axis=1,ignore_index=False)
print(all_combined_data['med_pos'])
all_combined_data.to_csv(my_dir+'all_combined_data_snap33')
'''

'''
combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.01].index)
col_names=(combined_data.columns)
print('low merger',len(combined_data['ID']))
combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']<0.01].index)
col_names=(combined_data.columns)
print('high merger ',len(combined_data['ID']))
'''     
