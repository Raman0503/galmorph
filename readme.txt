#########################################################################
The scripts in this folder  are based on code written by Junkai Zhang, Dr Stijn Wuyts and Raman Sharma. A functional structure rather than object orientated approach is taken. No classes are generated or used. Standard Python libraries as well as Astropy,Numpy,Astrodendro,photutils,Pandas are used

Only the following 3 scripts are needed for the time being

-mass_bulge_find.py- contains functions that interact with the Illustris database and generates catalogues and merger trees as well as generating data such as BT ratio and clumpiness indices. The generated catalogues are moved in CSV files and Analysis of the files as dataframes is also done using the executable scripts below the functions.

-hdf_reader.py- contains functions that interact with the IllustrisTNG visualiser. This also contains functions that calculate the clumpiness indices from the image as well as ellipticity.

-merger_tree.py- contains functions that interact with the merger tree. This was written by Junkai to resolve an issue with finding descendants in the merger tree

http_get, merger_time and redshift_distance.py are dependant scripts and called at various points 

-------------------------------------------------------------------------
The following scripts are works in progress

particle_reader reads particle data from TNG-50. Can be used to develop analysis of 3D structure of clumps

fits_reader interacts with fits files to open them and read data

random_forest is a supervised ML script that can generate datasets as well as train and develop algorithm that determines most important factors leading to outcome.  
