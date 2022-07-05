#RS code to open hdf files, get hdf files from visualiser and do blob detection, CAS indicies and dendrogram. Can also do ellipticity. Also added lis reader 
#Can use this as a git test
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import deepdish as dd
from astropy.table import Table
from astropy.io import ascii
from astropy.convolution import Gaussian2DKernel,convolve,convolve_fft,interpolate_replace_nans
from scipy import ndimage,misc
from photutils.isophote import EllipseGeometry,Ellipse
import photutils.aperture as phot_ap
from photutils.aperture import EllipticalAperture,CircularAperture,CircularAnnulus,BoundingBox,aperture_photometry,ApertureMask
#from photutils.aperture import ApertureStats
#from photutils.segmentation import detect_threshold
from astrodendro import Dendrogram,Structure
from skimage.feature import blob_log
from skimage import exposure
import requests
import pandas as pd
import seaborn as sns

dir_TNG='/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG'
my_imgs='/home/AstroPhysics-Shared/Sharma/Illustris/images/'
my_dir='/home/AstroPhysics-Shared/Sharma/Illustris/data_output/'
plot_dir='/home/AstroPhysics-Shared/Sharma/Illustris/plots/'
output_dir=dir_TNG+'/bulgegrowth/output/'
data_dir=dir_TNG+'/bulgegrowth/output/'
#convention in these hdfs is header and grid as paths

def legacy_read_hdf(filename,key_num):
    #get the keys from a hdf file and extract data from any key number. returns all keys and data for a key in a list. 
    with h5py.File(filename,'r') as f:
        all_keys=(list(f.keys()))
        #print(all_keys)
        a_group_key=list(f.keys())[key_num]
        data=np.array(f[a_group_key])
        return(all_keys,data)

def read_hdf(filename):
    d=dd.io.load(filename)
    return(d)
    
def show_hdf(filename,path='grid'):
    #show the file image assuming grid holds image data. Specific to Illustris as grid is the name of the image. 
    f=h5py.File(filename,'r')
    stellar_pos=f['grid'][()]
    original_image=plt.imshow(stellar_pos)
    original_image.set_cmap('plasma')
    plt.title('Image of HDF file')
    plt.colorbar(label='Hapha map')
    plt.show()

def read_lis(filename):
    list_file=ascii.read(filename)
    #ascii.read produces a table
    col_list=list_file.colnames
    #print(col_list)
    return(col_list,list_file)
    
    
#-------End hdf file handling-------------------------------------------------------

def get_clump_index(array,xsize,ysize,R_pet,blur_size):
#This is the CAS index.  Need to get linear h-alpha image from visualiser first. Need to provide total R flux from somewhere. Is then used to calculate gaussian. Can use a petrosian from the log image then.  
#use a quarter of the petrosian as gaussian and centre removal size like Whitney et al.  
    R_pix=int(0.2*R_pet)
    cent_pos=int((xsize+ysize)/4)
    blurring=Gaussian2DKernel(blur_size)
    gauss=Gaussian2DKernel(R_pix)
    array=convolve_fft(array,blurring)
#apply the kernel    
    convolved_img=convolve_fft(array,gauss)
    print('convolution complete')
#apply the CAS computation to the image and get S    
    Cluster_img=(array-convolved_img)
    Cluster_img[Cluster_img<0]=0
#remove the centre of the array and image before calculating S
    Cluster_img[cent_pos-int(0.5*R_pix):cent_pos+int(0.5*R_pix),cent_pos-int(0.5*R_pix):cent_pos+int(0.5*R_pix)]=0
    array[cent_pos-int(0.5*R_pix):cent_pos+int(0.5*R_pix),cent_pos-int(0.5*R_pix):cent_pos+int(0.5*R_pix)]=0
    S=np.sum(Cluster_img)/np.sum(array)
    print('clump index is',S)
    return(S)

#------end CAS clump index----------------------------------------------------


def get_hdf(snapnum,subHaloID,nPixels,rotation=None,axes=None,size=5,partField=None,partType=None,sim_name='TNG50-1'):
# make HTTP GET request to path. Gets HDF5 file from TNG visualiser. Replace API and params with whatever you need  
#my API key. axes is 0,1 viewing in x,y plane along z axis. Rotation is specific to galaxy and takes options edge-on and face-on. partType is stars or gas. Partfield is halpha. use 5 times half stellar radius size of image at 200 pixel size 
    baseUrl ='http://www.tng-project.org/api/'
    params={'partField':partField,'partType':partType,'ctName':'plasma','nPixels':nPixels,'rasterPox':1100,'plotStyle':'edged','sizeType':'rHalfMassStars','size':size,'axes':axes,'rotation':rotation}
    headers = {"api-key":"048c9d870bdb6e245f533249a0d69210"}
    path=baseUrl+sim_name+'/snapshots/'+str(snapnum)+'/subhalos/'+str(subHaloID)+'/vis.hdf5'
    r = requests.get(path,params=params,headers=headers)
    #print(r)
    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    #save data to my imgs folder
    if 'content-disposition' in r.headers:
        filename = my_imgs+r.headers['content-disposition'].split("filename=")[1]
        print(f'sucessfully got {filename}')
    with open(filename,'wb') as f:
        f.write(r.content)
    print(filename)
    return(filename) # return the filename string
    
#end get_hdf---------------------------------------------                       

def get_lin_array(file_name):
#Get linear valued array from filename assuming Halpha image as flux is in log10 units. Returns the linear and cleaned log image    
    f=h5py.File(file_name,'r')
    print('Opened file')
    #turn the identified path into np array
    stellar_pos=f['grid'][()]
    #turn nans to 0
    stellar_pos_orig=np.nan_to_num(stellar_pos,nan=35)
    #reduce the exponent
    stellar_pos=stellar_pos_orig-35
    #linearise array 
    stellar_pos_lin=10**(stellar_pos)
    return(stellar_pos_lin,stellar_pos_orig)

def get_flux_array(filename):#gets flux form h5 file if  values are in abs mag
    f=h5py.File(filename,'r')
    print('Opened file')
    #turn the identified path into np array
    stellar_pos=f['grid'][()]
    #stellar_pos_orig=np.nan_to_num(stellar_pos,nan=0)
    stellar_flux=10**(stellar_pos/(-2.5))
    return(stellar_flux)
    
#----end lin array--------------------------------------------------------------------------------

def get_petrosian(array,xsize,ysize):
#gets petrosian from array eta=0.2. If petrosian falls outside image - image pixel radius-dr is default radius.     
    centrex=int(xsize/2)
    centrey=int(ysize/2)
    position=(centrex,centrey)
    R=int((xsize+ysize)/4)
    dr=2
    eta=0.2
    n_radii=int(R/dr)
    #print('number of radii',n_radii)
    for i in range(1,n_radii):
        r=i*dr
        inner_ap=CircularAperture(position,r=r)
        outer_ap=CircularAperture(position,r=r+dr)
        phot_inner=aperture_photometry(array,inner_ap)
        phot_outer=aperture_photometry(array,outer_ap)
        inner_sum=phot_inner['aperture_sum']
        outer_sum=phot_outer['aperture_sum']
        sum_at_r=outer_sum-inner_sum
        area_at_r=(np.pi*(r+dr)**2)-(np.pi*(r)**2)
        I_per_pix=(sum_at_r)/(area_at_r)
        av_I=inner_sum/(np.pi*r**2)
        if (I_per_pix/av_I)<=eta:
            r_pet=r
            break
        else:
            r_pet=r
    print('petrosian radius is',r_pet)
    return(r_pet)



#-----------end get petrosian--------------------------------------- 

def altern_get_petrosian(mass,xsize):
    '''Uses Whitney 2019 paper table 3 to get m and c and then retruns petrosian'''
    R_pet=10**((0.14*mass)-0.7)
    
   

def get_ellipticity(array,xsize,ysize,sma,eps,pa):
#uses photoutils to define ellipticity only use on linear array pa in degrees. Edited Jan 21 to smooth the array first and slowly increase image change untill ellipticity converges
    for sma in range(10,100,10):
        for pa in range(15,180,30):
            geometry=EllipseGeometry(x0=xsize/2,y0=ysize/2,sma=sma,eps=eps,pa=(pa/180)*np.pi)
            print('fitting sma ',sma)
            for sigma in range(2,20):
                gauss=Gaussian2DKernel(sigma)
                print('fitting gaussian size',sigma)
#apply the kernel
                smooth_array=convolve_fft(array,gauss)
                ellipse=Ellipse(smooth_array,geometry)
                isolist=ellipse.fit_image(minit=20,maxit=100,sclip=2,nclip=3)
                elliptics=isolist.eps
                final_sma=isolist.sma
                final_pa=isolist.pa
                e_error=isolist.ellip_err
                flux=isolist.tflux_e
    #intensity_along_path=isolist.intens
#Rather than using C80, trying to use total flux. Went back to C80 for Jan test
                try:
                    e=elliptics[-1]
                    total_flux=flux[-1]
                    C80_flux=0.8*total_flux
                    C80_pos=0
                    for i in range(len(elliptics)):
                        if flux[i]>C80_flux:
                            C80_pos=i
                            break
                    e=elliptics[C80_pos]
                    sma=final_sma[C80_pos]
                    pa=final_pa[C80_pos]
                    break
                except IndexError:
                    e=1
                    continue
            try:
                e=elliptics[-1]
                total_flux=flux[-1]
                C80_flux=0.8*total_flux
                C80_pos=0
                for i in range(len(elliptics)):
                    if flux[i]>C80_flux:
                        C80_pos=i
                        break
                e=elliptics[C80_pos]
                sma=final_sma[C80_pos]
                pa=final_pa[C80_pos]
                break
            except IndexError:
                e=1
                continue
        try:
            e=elliptics[-1]
            total_flux=flux[-1]
            C80_flux=0.8*total_flux
            C80_pos=0
            for i in range(len(elliptics)):
                if flux[i]>C80_flux:
                    C80_pos=i
                    break
            e=elliptics[C80_pos]
            sma=final_sma[C80_pos]
            pa=final_pa[C80_pos]
            break
        except IndexError:
            e=1
            continue
    #print(isolist.to_table()) 
    #original_image=plt.imshow(array)
    #original_image.set_cmap('plasma')
    #plt.title('Image of HDF file')
    #plt.colorbar(label='Hapha map')
    #geometry=EllipseGeometry(x0=xsize/2,y0=ysize/2,sma=sma,eps=e,pa=pa)
    #aper=EllipticalAperture((geometry.x0,geometry.y0),geometry.sma,geometry.sma*(1-geometry.eps),geometry.pa)
    #aper.plot(color='white')
    #plt.show()

    #final_e_error=e_error[-1]
    #e_error=e_error[C80_pos]
    print('the e value is', e)
    return(e)

#-------end ellipticity----------------------------------------------------------------
   
def get_CI(array,xsize,ysize,R):
#gets CAS concentration index. Need to provide R at total flux     
    centrex=int(xsize/2)
    centrey=int(ysize/2)
    position=(centrex,centrey)
    R_img=(xsize+ysize)/4
    dr=2
    eta=0.2
    n_radii=int(R_img/dr)
    total_ap=CircularAperture(position,r=R)
    total_phot=aperture_photometry(array,total_ap)
    total_flux=total_phot['aperture_sum']
    C80=1
    C20=1
    #get C20 and C80 radii from curve of growth
    for i in range(1,n_radii):
        r_C=i*dr
        ap=CircularAperture(position,r_C)
        ap_phot=aperture_photometry(array,ap)
        ap_flux=ap_phot['aperture_sum']
        if (ap_flux/total_flux)<=0.2:
            C20=r_C
        if (ap_flux/total_flux)<=0.8:
            C80=r_C
        else:
            continue
    print('radii are at c20 and C80 ',C20,C80)
    CI=5*np.log(C80/C20)
    print('Concentration index is',CI)
    return(CI)

#---------------end CAS concentration index--------------------------------------

def get_asymm(array,xsize,ysize):
#Gets assymmetry index but assumes linear array    
    img_orig=array
    imag_rot=ndimage.rotate(img_orig,180,reshape=False)
    assym=np.sum(np.abs(img_orig-imag_rot))/np.sum(img_orig)
    print('Assymetry index is',assym)
    return(assym)

#---------------end CAS aymmetry index---------------------------------------------

    
def get_clumps(image):
#This is from Lenkic et al 2021. Uses skimage blob_log on any image to identify position(pixel) and radius 
#apply blob detection
    #image=exposure.equalize_hist(image)
    clumps=blob_log(image,max_sigma=20,min_sigma=5,num_sigma=200,threshold=0.1)
    print(clumps)
    return(clumps)

#---------end blob_log-------------------------------------------

def get_clumps_img(image,r_pet,x_size,y_size,blur_size):
#use conselice method to get the image but then cut 5SD above galaxy pixels.Added a blur to original image to make sure clumps are above resolution limit.Blur size is FWHM=1.5kpc or sigma 0.637. Use (40/half_rad)*0.637 
    blurring=Gaussian2DKernel(blur_size)
    gauss=Gaussian2DKernel(0.2*r_pet)
    clump_img=np.empty((y_size,x_size))
#apply the kernel    
    blurred_image=convolve_fft(image,blurring)
    smooth_img=convolve_fft(blurred_image,gauss)
    print('Image smoothed')
    residual=np.array(blurred_image-smooth_img)
    residual[residual<0]=0
    st_dev=get_galaxy_sd(residual,x_size,y_size,r_pet)
        #smoothed_img=plt.imshow(smooth_img,cmap='viridis')
    print('st dev of image is',st_dev)
#apply clump detection by simple cut of light more than 5 sd.  
    for x in range(0,x_size):
        for y in range(0,y_size):
            if residual[y,x]>5*st_dev:
                clump_img[y,x]=residual[y,x]
            else:
                clump_img[y,x]=0
    return(clump_img,residual,blurred_image,smooth_img)
    
#---------end get clump img----------------------------------------------

def get_clump_mask(image,clumps,x_size,y_size): 
    #get a masked image of clumps=0. Uses identified position in clumps isolist generated from residual
    masked_image=image
    for pos in range(len(clumps[:,])):
        #for each clump- take the photometry from the aperture then mask all clumps and do annular photometry for background. Take background away from clumps and collect clumps together to get flux.  
        position=((clumps[pos,1],clumps[pos,0]))
        radius=(clumps[pos,2]*np.sqrt(2))
        ap=CircularAperture(position,radius)
        #mask is 1 everything else is 0. standard pixel size
        mask=ap.to_mask(method='center')
        #applies the 1 and 0 to an image the same size as original
        masks=mask.to_image(shape=((x_size,y_size)))
        #now times by the image to get just the masked pixels.
        new_image=masks*image
        masked_image=masked_image-new_image
    return(masked_image)
#-----------------end get clump mask-------------------------- 

def get_clump_light(image,clumps,masked_image,r_pet):
#Takes the flux from clumps and removes background disk flux using masked image. Then take ratio of clump flux to total image flux as  an index. Use an image with linear values, not log values.gal radius is half mass rad to help exclude blobs less than 1kpc in.    
    total_clump=np.array([])
    for pos in range(len(clumps[:,])):
#for each clump- take the photometry from the aperture then mask all clumps and do annular photometry on fully masked image for background. Take background away from clumps and collect clumps together to get flux.  
        position=((clumps[pos,1],clumps[pos,0]))
#multiply clump pixel radius by sqrt2 as returns fwhm.          
        radius=(clumps[pos,2]*np.sqrt(2))
        ap=CircularAperture(position,radius)
        clump_flux=aperture_photometry(image,ap)
        flux=clump_flux['aperture_sum']
        #print(flux)
        annulus_aperture=CircularAnnulus(position,r_in=radius*4,r_out=radius*5)
        annulus_flux=aperture_photometry(masked_image,annulus_aperture)
        ann_flux=annulus_flux['aperture_sum']
        bkg_flux=ann_flux/((np.pi*(5*radius)**2)-(np.pi*(4*radius)**2))
        total_bkg_flux=bkg_flux*(np.pi*radius**2)
        adjusted_flux=flux-total_bkg_flux
        #print(adjusted_flux)
        gal_r=np.sqrt((100-position[0])**2+(100-position[1])**2)
        #just use clumps that have position >1kpc from centre
        if gal_r>(0.1*r_pet):
            total_clump=np.append(total_clump,adjusted_flux)
        #ap.plot(color='white')
        #remove central region of image as central clumps are also excluded
    image[int(100-0.1*r_pet):int(100+0.1*r_pet),int(100-0.1*r_pet):int(100+0.1*r_pet)]=0
    total_clump_flux=np.sum(total_clump)
    total_flux=np.sum(image)
    clump_index=total_clump_flux/total_flux
    #clump_plot=plt.imshow(image,cmap='viridis')
    #plt.colorbar(label='halpha map')
    #plt.show()
    return(clump_index)

#-----end clump flux known as blob index---------------------------------------

def get_galaxy_sd(image,y_size,x_size,r_pet):
#get just galaxy pixels by cutting off those that are a fraction of max. 0.0001 chosen by experimentation empirically. Identifies galaxy pixels. 
    img_sum=np.array([])
    cent_pos=int((y_size+x_size)/4)
    r=int(0.1*r_pet)
    image[cent_pos+r:cent_pos-r,cent_pos+r:cent_pos-r]=0
    for y in range(0,y_size):
        for x in range(0,x_size):
            if image[y,x]>0:
                img_sum=np.append(img_sum,image[y,x])
            else:
                image[y,x]=0
    st_dev=np.std(img_sum)
    min_img=np.min(image)
    max_img=np.max(image)
    return(st_dev)

#----------end galaxy sd-------------------------------------------------------------

#uses dendrogram method to get dendrogram and views. smooths according to r_pet again See Meng 2020. 
def get_dendro(array,radius,x_size,y_size,blur):
    blurring=Gaussian2DKernel(blur)
    blurred_img=convolve_fft(array,blurring)
    gauss=Gaussian2DKernel(0.2*radius)
    #apply the kernel    
    smooth_img=convolve_fft(blurred_img,gauss)
    print('Image smoothed')
    residual=np.array(blurred_img-smooth_img)
    residual[residual<0]=0
    sd=get_galaxy_sd(residual,x_size,y_size,radius)
    d=Dendrogram.compute(residual,min_value=5*sd,min_delta=sd,min_npix=5)
    #v=d.viewer()
    #v.show()
    #print(d.trunk)
    return(d,blurred_img)#return the blurred img for leaf flux
#-------------end get dendrogram----------------------------------------------------- 

#uses leaf structures to get masks and then pixel values. No background removal as cannot get pixel position of leaves(as far as I have worked out yet).Can get pix pos but instead remove central 1kpc  
def get_leaf_flux(array,dendrogram,r_pet):
    print('number of leaves is ', len(dendrogram.leaves))
    leaf_sum=([])
    for leaf in dendrogram.leaves:
        mask=leaf.get_mask()
        masked_image=mask*array
        masked_image[100-int(0.1*r_pet):100+int(0.1*r_pet),100-int(0.1*r_pet):100+int(0.1*r_pet)]=0
        clump_flux=np.sum(masked_image)
        leaf_sum=np.append(leaf_sum,clump_flux)
    dendro_clump_index=(np.sum(leaf_sum)/np.sum(array))
    return(dendro_clump_index)

def get_leaf_indices(dendrogram):
    nleaves=len(dendrogram.leaves)
    print(f'Number of leaves is {nleaves}')
    pix_pos=([])
    for i,leaf in enumerate(dendrogram.leaves):
        idx=leaf.indices()
        npix=leaf.get_npix()
        vals=leaf.values()
        peak=leaf.get_peak()
        pix_pos=np.append(pix_pos,idx)
        #print(f'indices for leaf {i}  are{idx} and the number of pixels is {npix} withe a peak of {peak}. The Values in the pixels are {vals}')
    return(pix_pos,npix,nleaves)    
        
#use fft to get amplitude frequancy space from anuli of progressively increasing radii.NOTE THE INDEXING OF THIS FUNCTION IS COMPLETELY WRONG DO NOT USE   
def get_structure_legacy(array,r,dr,theta):
    '''uses fft to plot structures normalised against intensity. r in pixels. dr in pix, theta in number of degrees deltas per 2*pi'''
    all_Amp=np.array([])
    all_r=([])
    all_ampm2=np.array([])
#fig,ax=plt.subplots()
    for i in range(dr,r,dr):
        all_pix=np.array([])
        all_angles=np.array([])
        sep_all_pix=np.array([])
        for j in range(theta):
            #print(j)
            bar_img=ndimage.rotate(array,j*(360/theta),reshape=False)
            pix_value=(np.sum(bar_img[i:i+2,99:101]))/4
            #print(pix_value)
            all_pix=np.append(all_pix,pix_value)
            dtheta=(j/theta)*2*np.pi
            all_angles=np.append(all_angles,dtheta)
        for k in range(int(0.5*(len(all_pix)))):#sample the annulus every 180 degrees to extract the m2 frequency. Needs to be half the length of theta 
            sep_all_pix=np.append(sep_all_pix,all_pix[k])
            sep_all_pix=np.append(sep_all_pix,all_pix[k+int(0.5*theta)])
        fft_array=np.fft.fft(sep_all_pix)
        n=sep_all_pix.size
        delta_theta=180#*np.pi*2  separation in degrees. 
        freq=np.fft.fftfreq(n,d=delta_theta)
        ampm2=(np.sqrt((fft_array.real[2]**2)+(fft_array.real[-2])**2))/np.sum(sep_all_pix)
        all_ampm2=np.append(all_ampm2,ampm2)
        #print('frequency',len(freq),freq)
        #print(freq[0:5],freq[-5:-1])
        #fig,ax=plt.subplots()
        #ax.plot(freq,fft_array.real,color='blue',ls='-',lw=0.5)
        #plt.show()
        amp=np.sqrt(np.sum((fft_array.real)**2))/np.sum(sep_all_pix)
        print(amp,ampm2)
        all_Amp=np.append(all_Amp,amp)
        all_r=np.append(all_r,i)
    return(all_Amp,all_r,all_ampm2)

def get_bar(array,r,dr,dtheta):#dtheta in degrees. Using rosas method and not rotating image but sampling image directly. Get max A2 only in range where phi is constant doesnt work but getting A2 in Re seems to be better
    A2_vals=np.array([])
    phi_vals=np.array([])
    all_r=np.array([])
    for i in range(dr,r,dr):
        all_pix=np.array([])
        all_m2=np.array([])
        all_sin_angles=np.array([])
        all_cos_angles=np.array([])
        for t in np.arange(0,2*np.pi,(dtheta/360)*2*np.pi):
            x_val=100+int(i*np.sin(t))
            y_val=100+int(i*np.cos(t))
            m2=(array[y_val:y_val+1,x_val:x_val+1]/4)*np.exp(2j*t)
            pix_val=array[y_val:y_val+1,x_val:x_val+1]/4
            all_pix=np.append(all_pix,pix_val)
            all_m2=np.append(all_m2,m2)
            sin_angle=(array[y_val:y_val+1,x_val:x_val+1]/4)*np.sin(2*t)
            cos_angle=(array[y_val:y_val+1,x_val:x_val+1]/4)*np.cos(2*t)
            all_sin_angles=np.append(all_sin_angles,sin_angle)
            all_cos_angles=np.append(all_cos_angles,cos_angle)
        A2=np.abs(np.sum(all_m2))/np.sum(all_pix)
        phi=np.arctan(np.sum(all_sin_angles)/np.sum(all_cos_angles))
        A2_vals=np.append(A2_vals,A2)
        phi_vals=np.append(phi_vals,phi)
        all_r=np.append(all_r,i)
    true_A2=np.array([])
    true_len=np.array([])
    for k in range(len(phi_vals)):
        if 1.1>phi_vals[k]/phi_vals[k+1]>0.9:
            A2_true=A2_vals[k]
            true_A2=np.append(true_A2,A2_true)
            true_len=np.append(true_len,all_r[k])
        else:
            break
    try:
        final_A2=np.max(true_A2)
        max_idx=np.argsort(true_A2)
        final_len=true_len[max_idx[0]]
    except ValueError:
        final_A2=0
        final_len=0
    new_final_A2=np.max(A2_vals)
    max_idx=np.argsort(A2_vals)
    new_final_len=all_r[max_idx[0]]
    return(final_A2,final_len,A2_vals,phi_vals,all_r,new_final_A2,new_final_len)       
            
            

#----------------end get leaf index---------------------------------        
    

#-----Executables---------------------------

#h=read_hdf(dir_TNG+'/bulgegrowth/output/SFGMgt10_BTlt03_noM/gasprop/gas_Mgt10.0_SFG_BTlt03_noM_sn33.h5',0)
#print(h)
#for i in range(len(h[0])):
#    print(h[0][i])

'''
#idx 2 is summed mass ratio. 
#hdf=read_hdf(my_dir+'merger_Mgt9.0_All_sn33.h5',2)
#print(hdf[0][2])
#print(hdf[1])
#print(hdf[1].shape)
IDlist=read_lis(my_dir+'Mgt9.0_All_sn33.lis')
#print(IDlist[0])
print(IDlist[1])
'''
'''
#test some functions in hdf reader 
ID_new=6
init_snapnum=33
img=my_imgs+'grid_subhalo_TNG50-1_'+str(init_snapnum)+'_'+str(ID_new)+'.hdf5'
#print(filename)
array=get_lin_array(img)
#print(array)
original_image=array[0]
log_image=array[1]
radius=200
blur=4
d=get_dendro(original_image,radius,200,200,blur)
#flux=get_leaf_flux(original_image,d[0],4)
#masked_image=flux[1]
#plt.imshow(masked_image)
#plt.show()
p=get_leaf_indices(d[0])
all_pos=p[0]
npix=p[1]
ypos=all_pos[0:npix]
xpos=all_pos[npix-1:-1]
print(len(xpos),len(ypos))
leaf_image=np.empty((200,200))
#print(leaf_image.shape)
for y in ypos:
    #print(y)
    for x in xpos:
        #print(x)
        leaf_image[int(y),int(x)]=original_image[int(y),int(x)]
plt.imshow(leaf_image)
plt.show()
#plt.imshow(original_image,cmap='gray')
#plt.legend('Original')
#plt.colorbar()
#plt.show()
#E=get_ellipticity(original_image,100,100,10,0.5,45)
#clump_image=get_clumps_img(original_image,radius,200,200)
#plt.imshow(np.log10(clump_image))
#plt.show()
#clumps=get_clumps(clump_image)
#masked_image=get_clump_mask(original_image,clumps,200,200)
#plt.imshow(masked_image)
#plt.show()
#new_clump_index=get_clump_light(original_image,clumps,masked_image)
#print('new clump index is', new_clump_index)
#old_clump_index=get_clump_index(original_image,200,200,radius)
#print('old_clump index is',old_clump_index)
#print(Clump_index)
'''
'''
filename=my_dir+'gas_Mgt9.0_SFG_BTlt03_sn33.h5'
paths=read_hdf(filename,0)
print(paths[0])
for num in range(len(paths[0])):
    paths=read_hdf(filename,num)
    #print(paths[0][num],paths[1].shape)
    if paths[0][num]=='lMstar':
        print(paths[0][num],paths[1])
    if paths[0][num]=='delBT_thresh':
        print(paths[0][num],paths[1])
    else:
        continue
    #filename=my_dir+'gas_Mgt9.0_SFG_BTlt03_sn33.h5'
#paths=read_hdf(filename,0)
#print(paths[0])
#for num in range(len(paths[0])):
#    paths=read_hdf(filename,num)
#    print(paths[0][num],paths[1].shape)
#filename=my_dir+'merger_Mgt9.0_All_sn33.h5'
#paths=read_hdf(filename,0)
#print(paths[0])
#for num in range(len(paths[0])):
#    paths=read_hdf(filename,num)
#    print(paths[0][num],paths[1].shape)    

data=read_lis(my_dir+'Mgt9.0_SFG_BTlt03_sn33.lis')
#print(data.dtype)
print('lis data',len(data[1][data[0][0]]))
print('table length',len(data[1]))
'''

'''
#get leaves and get flux from leaves
if __name__=="__main__":
    filename=get_hdf(33,8072,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    #print(filename)
    array=get_lin_array(filename)
    #print(array)
    original_image=array[0]
    log_image=array[1]
    d=get_dendro(original_image,100,200,200)
    index=get_leaf_flux(original_image,d)
    print('clump index from leaves is ',index)
    #p=d.plotter()
    #for leaf in d.leaves:
    #    print('number of leaves is ', len(d.leaves))
        #p.plot_contour(ax,lw=2,colors='red')
    #    mask=leaf.get_mask()
    #    ax.imshow(mask)
    #plt.show()    

'''
'''
#non_bar_img=get_hdf(33,5,200,rotation='face-on',axes=None,size=5,partField='mass',partType='Stars',sim_name='TNG50-1')
non_bar_img=my_imgs+'grid_subhalo_TNG50-1_'+str(33)+'_'+str(5)+'.hdf5'
bar_img=get_hdf(33,8069,200,rotation='face-on',axes=None,size=5,partField='halpha',partType='gas',sim_name='TNG50-1')
bar_img=my_imgs+'grid_subhalo_TNG50-1_'+str(33)+'_'+str(8069)+'.hdf5'
non_bar_array=get_lin_array(non_bar_img)[0]
bar_array=get_lin_array(bar_img)[0]
#plt.imshow(get_lin_array(bar_img)[1])
#plt.show()
all_Amp=np.array([])
all_r=([])
#fig,ax=plt.subplots()
for i in range(2,98,2):
    all_pix=np.array([])
    all_angles=np.array([])
    for j in range(360):
        bar_img=ndimage.rotate(bar_array,j,reshape=False)
        pix_value=(np.sum(bar_img[100+i:100+i+2,100-1:100+1]))/4
        #print(f'av pix value at {j} is {pix_value}')
        all_pix=np.append(all_pix,pix_value)
        theta=(j/360)*2*np.pi
        #print(theta)
        all_angles=np.append(all_angles,theta)
    #print(all_pix.shape)
    #print(all_angles.shape)
    fft_array=np.fft.fft(all_pix)
    #sq_fft_array=((fft_array)**2)
    #print('fft length',len(fft_array))
    idx=np.argsort(fft_array.real)
    n=all_pix.size
    delta_theta=(1/360)*np.pi*2
    freq=np.fft.fftfreq(n,d=delta_theta)
    idx_max=idx[-2]
    freq_at_max=freq[idx_max]
    #print('Max frequency is ',freq_at_max)
    #print('frequency length',len(freq))
    #shifted_freq=np.fft.fftshift(freq)
    #print(len(shifted_freq))
    amp=np.sqrt(np.sum((fft_array.real)**2))/np.sum(all_pix)
    print(amp)
    #ax.plot(freq,fft_array.real,lw=0.5,color=(1/i,0,0))
    #ax.set_xlim(-4,4)
    all_Amp=np.append(all_Amp,amp)
    all_r=np.append(all_r,i)
    #print('fft array',fft_array)
#plt.show()
fig,ax=plt.subplots()
plt.plot(all_r,all_Amp,lw=0.5,ls='-',color='blue')
    #plt.show()
    #annulus=CircularAnnulus((100,100),r_in=i,r_out=i+2)
    #flux=aperture_photometry(bar_array,annulus)
    #print(flux)
    #plt.imshow(annulus_image)
    #plt.show()
    #fft_array=np.fft.rfft2(annulus_image)
    #freq=np.fft.fftfreq(annulus_image.shape[-1])
    #print(i,freq)
    #fig,ax=plt.subplots()
    #plt.plot(freq,fft_array,lw=0.5,color='blue',ls='-')
ax.set_xlabel('Radius')
ax.set_ylabel('fft amplitude')
plt.show()
'''

#non_bar_img=get_hdf(33,5,200,rotation='face-on',axes=None,size=5,partField='halpha',partType='gas',sim_name='TNG50-1')
combined_data=read_hdf(my_dir+'combined_data_at_snap40.h5')
print(combined_data.keys())
all_bars=np.array([])
for i,ID in enumerate(combined_data['subfindID']):
    #bar_img=my_imgs+'grid_subhalo_TNG50-1_'+str(33)+'_'+str(ID)+'.hdf5'
    bar_img=get_hdf(40,ID,200,rotation='face-on',axes=None,size=5,partField='stellarBand-jwst_f444w',partType='stars',sim_name='TNG50-1')
    #bar_img=get_hdf(40,ID,200,rotation='face-on',axes=None,size=5,partField='halpha',sim_name='TNG50-1')
    bar_array=get_flux_array(bar_img)
    #log_bar_array=2**get_lin_array(bar_img)[1]
    bar_A2=get_bar(bar_array,40,2,1)
    print(f'bar A2 value is {bar_A2[5]},length is {bar_A2[6]}  SWbar is',combined_data['SWbar'][i])
    #fig,ax=plt.subplots(ncols=2,nrows=2,sharex=False)
    #ax[0,0].plot(bar_A2[4],bar_A2[2],c='blue',lw=0.5,label='bar A2')
    #ax[0,0].plot(non_bar_A2[4],non_bar_A2[2],c='red',lw=0.5,label='Non bar A2')
    #ax[1,0].plot(bar_A2[4],bar_A2[3],c='blue',lw=0.5,label='bar A2')
    #ax[1,0].plot(non_bar_A2[4],non_bar_A2[3],c='red',lw=0.5,label='Non bar A2')
    #ax[0,0].set_ylabel('A2 val')
    #ax[1,0].set_ylabel('Phi val')
    #ax[1,0].set_xlabel('Radius')
    #ax[0,1].imshow(bar_array)
    #ax[1,1].imshow(non_bar_array)
    #plt.legend()
    #plt.show()
    all_bars=np.append(all_bars,bar_A2[5])
fig,ax=plt.subplots()
sns.regplot(combined_data['SWbar'],all_bars,color='blue',ax=ax)
x=np.linspace(0,max(combined_data['SWbar']),len(combined_data['SWbar']))
y=x
ax.plot(x,y,lw=0.5,color='k')
ax.set_xlabel('SWBar')
ax.set_ylabel('RSBar')
plt.show()
