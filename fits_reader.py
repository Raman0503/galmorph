import sys
import subprocess
import numpy as np
import os
import illustris_python as il
from http_get import get
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits,ascii
from astropy.io.fits import getdata,getheader
from astropy.wcs import WCS
import pandas as pd
import sewpy
import eazy
import jwst
import csv
import deepdish as dd
filters=('f115w','f150w','f200w','f277w','f356w','f444w')
home_dir='/home/rs2755/JWST/CEERS/'



def show_fits(fits_filename):
    '''opens fits files and spits out info. Have to define ext first but default is 0. Returns headers and data in ext as well as the open file to manipulate with file[0].header[head]'''   
    #hdul=fits.open(fits_filename)
    fits.info(fits_filename)
    fitsfile=fits.open(fits_filename)
    for hdu in range(len(fitsfile)):
        hdul=(fitsfile[hdu])
        headers=fitsfile[hdul].header
        for head in headers:
            print(hdu,head,fitsfile[hdu].header[head])
    #fits contain .headers and .data. Headers are 80 byts and contain  a keyword(str) value(any type) comment(str). Select header details as header object is dict. hdrs['SCI'].  Access directly with .header method or create hdr object

#get value of header keywords 
    
    

#make a fits with primary headerr and 1 data image. Comment must eb a string. Can append to this using hdul.append(fits.ImageHDU_2(data2))
def make_fits(data,comment,filename):
    hdr=fits.Header()#create header object
    hdr['COMMENT']=comment
    empty_primary=fits.PrimaryHDU(header=hdr)
    image_hdu=fits.ImageHDU(data)#create image hdu
    hdul=fits.HDUList([empty_primary,image_hdu])
    hdul.writeto(filename)
    return(hdul)


#Execute this------------------------------------------------------------------
#mat=matplotlib.get_backend()
#print(mat)

#fig.savefig(home_dir+'JWST/CEERS/test.png',dpi=300)
#filename=home_dir+'ceers5_'+filters[1]+'_i2d.fits'
'''
filename=home_dir+'hlsp_candels_hst_acs_egs-tot-30mas-section11_f606w_v1.0_drz.fits'
fits.info(filename)
fitsfile=fits.open(filename)
for hdu in range(len(fitsfile)):
    hdul=(fitsfile[hdu])
    print(hdul)
    headers=fitsfile[hdul].header
    for head in headers:
        print(hdu,head,fitsfile[hdu].header[head])
'''
'''
sew = sewpy.SEW(workdir=home_dir,params=["NUMBER","X_IMAGE", "Y_IMAGE", "FLUX_APER","THETA_IMAGE","A_IMAGE","B_IMAGE","FLAGS"],config={"DETECT_MINAREA":5, "PHOT_APERTURES":"100","PIXEL_SCALE":0.015},sexpath="sex")
filename=home_dir+'hlsp_candels_hst_acs_egs-tot-30mas-section11_f606w_v1.0_drz.fits'
fits.info(filename)
fitsfile=fits.open(filename)
data=fitsfile[0].data
fig,ax=plt.subplots()
plt.imshow(np.log(data),cmap='viridis')
plt.show()
sew(home_dir+'hlsp_candels_hst_acs_egs-tot-30mas-section11_f606w_v1.0_drz.fits')
'''
'''
#view image of source extracted files
test_filename=home_dir+'hlsp_candels_hst_acs_egs-tot-30mas-section11_f606w_v1.0_drz.cat.txt'
test_file=ascii.read(test_filename)
test_Xval=test_file['X_IMAGE']
test_Yval=test_file['Y_IMAGE']
print('N of OBJECTS',len(test_Xval))
fig,ax=plt.subplots()
seg_img=fits.open(home_dir+'hlsp_candels_hst_acs_egs-tot-30mas-section11_f606w_v1.0_drz.fits')
data=np.log(seg_img[0].data)
ax.imshow(data,cmap='viridis')
for pos in range(len(test_Xval)):
    print(pos,test_Xval[pos],test_Yval[pos])
    ellipse=Ellipse(xy=(test_Xval[pos],test_Yval[pos]),width=test_file['A_IMAGE'][pos],height=test_file['B_IMAGE'][pos],angle=test_file['THETA_IMAGE'][pos],color='k',ls='--',fill=False)
    ax.add_patch(ellipse)
plt.show()
'''

'''
fig,ax_list=plt.subplots(nrows=3,ncols=2,figsize=(12,8))
for i,ax in enumerate(ax_list.ravel()):
    filename=home_dir+'ceers5_'+filters[i]+'_i2d.fits'
    show_fits(filename)
    with fits.open(filename) as hdul:
        data=np.log(hdul['SCI'].data)
        hdrs=hdul['SCI'].header
        print(hdrs)
        ax.imshow(data,cmap='gray',label=filters[i])
plt.legend()
plt.show()
'''
'''
#sewpy uses params.txt in same folder
for i in range(0,3):
    filename=home_dir+'ceers5_'+filters[i]+'_i2d.fits'
    sew = sewpy.SEW(workdir=home_dir,params=["NUMBER","X_IMAGE", "Y_IMAGE", "FLUX_APER","THETA_IMAGE","A_IMAGE","B_IMAGE","FLAGS"],config={"DETECT_MINAREA":100, "PHOT_APERTURES":"100","PIXEL_SCALE":0.015},sexpath="sex")
#out = sew("image.fits")
#call(filename)    
#print(out["table"]) # This is an astropy table.
#print(sew)
    sew(filename)
for i in range(3,6):
    filename=home_dir+'ceers5_'+filters[i]+'_i2d.fits'
    sew = sewpy.SEW(workdir=home_dir,params=["NUMBER","X_IMAGE", "Y_IMAGE", "FLUX_APER","THETA_IMAGE","A_IMAGE","B_IMAGE","FLAGS"],config={"DETECT_MINAREA":100, "PHOT_APERTURES":"100","PIXEL_SCALE":0.03},sexpath="sex")
#out = sew("image.fits")
#call(filename)    
#print(out["table"]) # This is an astropy table.
#print(sew)
    sew(filename)    
'''   

'''
#ascii.read makes a qtable. Access columns with .columns method. 
true_filename=home_dir+'ceers5_'+filters[1]+'_cat.ecsv'
test_filename=home_dir+'ceers5_'+filters[1]+'_i2d.cat.txt'
true_file=ascii.read(true_filename)
test_file=ascii.read(test_filename)
true_R=(true_file['xcentroid']**2+true_file['ycentroid']**2)**0.5
test_R=(test_file['X_IMAGE']**2+test_file['Y_IMAGE']**2)**0.5
trueR=np.sort(true_R)
testR=np.sort(test_R)
print('true length',len(trueR))
print('test length',len(testR))
fig,ax=plt.subplots()
ax.scatter(trueR,testR,s=3,color='k')
ax.set_ylabel('My test values')
ax.set_xlabel('True values')
plt.show()
'''
'''
test_filename=home_dir+'ceers5_'+filters[1]+'_i2d.cat.txt'
test_file=ascii.read(test_filename)
test_Xval=test_file['X_IMAGE']
test_Yval=test_file['Y_IMAGE']
print('N of OBJECTS',len(test_Xval))
fig,ax=plt.subplots(nrows=2,ncols=1)
seg_img=fits.open(home_dir+'ceers5_'+filters[1]+'_segm.fits')
data=seg_img['SCI'].data
empty_img=np.zeros(data.shape)
ax[0].imshow(empty_img)
for pos in range(len(test_Xval)):
    print(pos,test_Xval[pos],test_Yval[pos])
    ellipse=Ellipse(xy=(test_Xval[pos],test_Yval[pos]),width=test_file['A_IMAGE'][pos],height=test_file['B_IMAGE'][pos],angle=test_file['THETA_IMAGE'][pos],color='blue',fill=True)
    ax[0].add_patch(ellipse)
ax[1].imshow(data,cmap='viridis')
plt.show()
'''
'''
filename=ascii.read(home_dir+'ceers5_'+filters[1]+'.cat.ecsv')
print(filename)
'''
'''
fits.info(home_dir+'ceers5_'+filters[1]+'_segm.fits')
'''

#result=subprocess.run(["/home/rs2755/JWST/CEERS/galfitm-1.4.4-linux-x86_64.1","-c"],input="/home/rs2755/JWST/Galfit/ngcexamplesersic.txt",capture_output=True,text=True, timeout=10,check=True)
#print('result is',result.stdout)
#take source extractor output and then use that to identify sources and take 200 by 200 cutouts around the source.Turn them into a h5 file for coords and fits files for each object. Each obj number may refer to a different obj however. Need to rewrite to produce the same object fro each object number.    
'''
#print('shape is',data.shape)
#cutout_dict={}
for i,filter_name in enumerate(filters):
    all_coords=np.array([])
    true_filename=home_dir+'ceers5_'+filters[i]+'_i2d.cat.txt'
    SAM_cat=ascii.read(home_dir+"CEERS_SAM_input.cat")
    print(SAM_cat.columns)
    true_file=ascii.read(true_filename)
    print(true_file.columns)
    #true_R=(true_file['xcentroid']**2+true_file['ycentroid']**2)**0.5
    fig,ax=plt.subplots()
    print(len(true_file['X_IMAGE']))
    filename=home_dir+'ceers5_'+filters[i]+'_i2d.fits'
    image=fits.open(filename)
    data=image['SCI'].data
    header=fits.getheader(filename,ext=1)
    #access the ra, dec from x and y pixel coordinate
    wcs=WCS(header=header)
    for obj in range(0,10):#len(true_file['X_IMAGE'])):
        xpix=int(true_file['X_IMAGE'][obj])
        ypix=int(true_file['Y_IMAGE'][obj])
        print("coords",xpix,ypix)
        coord=wcs.all_pix2world(xpix,ypix,1)
        print('real coords are',coord)
        all_coords=np.append(all_coords,coord)
        #print('label',true_file['label'][obj])
        test_img=np.zeros((200,200))
        for x in range(0,200):
            for y in range(0,200):
                #print(x,y)
                try:
                    test_img[y,x]=data[ypix-100+y,xpix-100+x]
                except IndexError:
                    continue
        make_fits(test_img,'This is cutout from ceers 5 '+filter_name+str(obj),"ceers_"+filter_name+str(obj)+".fits")
        #cutout_dict.update({filter_name:all_coords})
        #cutout_dict.update({filter_name+str(obj):test_img})
        #plt.imshow(test_img,cmap='viridis')
        #fig.savefig("ceers_cutout_"+str(obj)+filter_name,format='png')
        print(str(obj),filter_name,'saved')
    #plt.show()
#dd.io.save('all_cutouts.h5',cutout_dict)
'''


#open fits image
fig,ax_list=plt.subplots(nrows=3,ncols=2)
for i,ax in enumerate(ax_list.ravel()):
    show_fits(home_dir+'ceers_'+filters[i]+'5.fits')
    image=fits.open(home_dir+'ceers_'+filters[i]+'5.fits')
    data=image[1].data
    ax.imshow(data,cmap='viridis')
plt.show()

'''
all_coords=dd.io.load('all_cutouts.h5')
print(all_coords.keys())
for i,filter in enumerate(filters):
    coord=all_coords[filter]
    coord=np.reshape(coord,(10,2))
    coord=np.sort(coord,0)
    print(coord)
'''   
'''
#print(coord)
#take source extractor output (pixel coords of centroid only) from 1 filter of mosaic  and then use that to identify same sources in all other filters and take 200 by 200 cutouts around the source.Turn them into a h5 for coords(ra,dec) and fits files for each object.
#print('shape is',data.shape)
coords_f115w_dict={}
all_coords=np.array([])
true_filename=home_dir+'ceers5_'+filters[0]+'_i2d.cat.txt'
SAM_cat=ascii.read(home_dir+"CEERS_SAM_input.cat")
print(SAM_cat.columns)
true_file=ascii.read(true_filename)
print(true_file.columns)
#true_R=(true_file['xcentroid']**2+true_file['ycentroid']**2)**0.5
filename=home_dir+'ceers5_'+filters[0]+'_i2d.fits'
image=fits.open(filename)
data=image['SCI'].data
header=fits.getheader(filename,ext=1)
#access the ra, dec from x and y pixel coordinate
wcs=WCS(header=header)
for obj in range(0,10):#len(true_file['X_IMAGE'])):
    xpix=int(true_file['X_IMAGE'][obj])
    ypix=int(true_file['Y_IMAGE'][obj])
    print("coords",xpix,ypix)
    coord=wcs.all_pix2world(xpix,ypix,1)
    print('real coords are',coord)
    all_coords=np.append(all_coords,coord)
    #print('label',true_file['label'][obj])
    test_img=np.zeros((200,200))
    for x in range(0,200):
        for y in range(0,200):
            #print(x,y)
            try:
                test_img[y,x]=data[ypix-100+y,xpix-100+x]
            except IndexError:
                continue
    make_fits(test_img,'This is cutout from ceers 5 '+filters[0]+str(obj),'ceers_'+filters[0]+str(obj)+'.fits')
all_coords=np.reshape(all_coords,(10,2))
coords_f115w_dict.update({filters[0]:all_coords})
dd.io.save('f115_coords.h5',coords_f115w_dict)
for i in range(1,6):
    filename=home_dir+'ceers5_'+filters[i]+'_i2d.fits'
    header=fits.getheader(filename,ext=1)
    #access the ra, dec from x and y pixel coordinate
    wcs=WCS(header=header)
    image=fits.open(filename)
    data=image['SCI'].data
    for j,coords in enumerate(all_coords):#len(true_file['X_IMAGE'])):
        print('checking coords',coords)
        #translate world coords into pix coords
        xpix=int(wcs.all_world2pix(coords[0],coords[1],1)[0])
        ypix=int(wcs.all_world2pix(coords[0],coords[1],1)[1])
        print("coords",xpix,ypix)
        #print('label',true_file['label'][obj])
        test_img=np.zeros((200,200))
        for x in range(0,200):
            for y in range(0,200):
                #print(x,y)
                try:
                    test_img[y,x]=data[ypix-100+y,xpix-100+x]
                except IndexError:
                    continue
        make_fits(test_img,'This is cutout from ceers 5 '+filters[i]+str(j),"ceers_"+filters[i]+str(j)+".fits")
        #cutout_dict.update({filter_name:all_coords})
        #cutout_dict.update({filter_name+str(obj):test_img})
        #plt.imshow(test_img,cmap='viridis')
        #fig.savefig("ceers_cutout_"+str(obj)+filter_name,format='png')
        print(str(j),filters[i],'saved')
    #plt.show()
#dd.io.save('all_cutouts.h5',cutout_dict)
'''
