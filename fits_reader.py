import sys
import numpy as np
import os
import illustris_python as il
from http_get import get
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits,ascii
from astropy.io.fits import getdata,getheader
import pandas as pd
import sewpy
import eazy
import jwst
import csv
filters=('f115w','f150w','f200w','f277w','f356w','f444w')
home_dir='/home/rs2755/JWST/CEERS/'



def show_fits(fits_filename):
    '''opens fits files and spits out info. Have to define ext first but default is 0. Returns headers and data in ext as well as the open file to manipulate with file[0].header[head]'''   
    #hdul=fits.open(fits_filename)
    fits.info(fits_filename)
    #fits contain .headers and .data. Headers are 80 byts and contain  a keyword(str) value(any type) comment(str). Select header details as header object is dict. hdrs['SCI'].  Access directly with .header method or create hdr object

#get value of header keywords 
    
    

#make a fits with primary headerr and 1 data image. Can append to this using hdul.append(fits.ImageHDU_2(data2))
def make_fits(data,hdr,filename):
    hdu=fits.PrimaryHDU(header=hdr)
    image_hdu=fits.ImageHDU(data)
    hdul=fits.HDUlists([hdu,image_hdu])
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
