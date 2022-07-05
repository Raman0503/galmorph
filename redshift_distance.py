from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.coordinates import Distance

# proper kpc
def distance(redshift, H0=70, Om0=0.3, Tcmb0=2.725):
    cosmo = FlatLambdaCDM(H0, Om0, Tcmb0)
    d = cosmo.kpc_proper_per_arcmin(redshift)
    return d.value

def lumonsity_distance(redshift, H0=70, Om0=0.3, Tcmb0=2.725):
    cosmo = FlatLambdaCDM(H0, Om0, Tcmb0)
    d = Distance(z=redshift, cosmology=cosmo)
    return d

def comoving_volume(z, H0=70, Om0=0.3, Tcmb0=2.725):
    cosmo = FlatLambdaCDM(H0, Om0, Tcmb0)
    #print(cosmo.comoving_volume(z+dz)-cosmo.comoving_volume(z-dz))
    return cosmo.comoving_volume(z).value

def separation(A,B):
    ra_A, dec_A = A
    ra_B, dec_B = B
    coord_A = SkyCoord(ra_A, dec_A, unit='deg')
    coord_B = SkyCoord(ra_B, dec_B, unit='deg')
    separation = np.array(coord_A.separation(coord_B))
    return separation

def Universe_age(redshift, H0=70, Om0=0.3, Tcmb0=2.725):
    cosmo = FlatLambdaCDM(H0, Om0, Tcmb0)
    age = cosmo.age(redshift)
    print(age)
    return age

if __name__ == "__main__":
   comoving_volume(2)
   #Universe_age([1,2,3])
   #D = lumonisty_distance(5)
   #print(D)
   #d5 = Distance(z=0.23, cosmology=WMAP5)
   #print(d5)

