#From Junkai - RS has edited to get more parameters from tree

import illustris_python as il
import numpy as np
from http_get import get
import h5py
import os
import pandas as pd
import time

def merger_tree(z=0):

    timelist = [time.time()]

    basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'
    
    # find the z corresponding snapshot number 
    
    baseUrl = 'http://www.tng-project.org/api/'

    r = get(baseUrl)

    names = [sim['name'] for sim in r['simulations']]
    i = names.index('TNG100-1')

    sim = get( r['simulations'][i]['url'] )

    snap = get( sim['snapshots'] )

    def find_snap_ID(z):
        select = 0
        for i in range(len(snap)):
            if abs(snap[i]['redshift']-z)<=0.1 and abs(snap[i]['redshift']-z)<abs(snap[select]['redshift']-z):
               select = i
        if abs(snap[select]['redshift']-z)>0.1:
           print('wrong redshift input')
           return
        return select

    select = find_snap_ID(z)
    snap_ID = snap[select]['number']

    
    # read the merger tree


    #fields = ['SubfindID', 'SnapNum', 'SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID', 'SubhaloMassType']


    #GroupFirstSub = il.groupcat.loadHalos(basePath,snap_ID,fields=['GroupFirstSub'])
    
    #print(GroupFirstSub,len(GroupFirstSub))

    #select = np.where(GroupFirstSub != -1)
    #GroupFirstSub = GroupFirstSub[select]

    #print('GroupFirstSub:', GroupFirstSub, len(GroupFirstSub))

    fields = ['SubfindID', 'SnapNum', 'SubhaloID']
    
    address = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/postprocessing/trees/tree/'
    

    offsetFile = il.groupcat.offsetPath(basePath, snap_ID)
    prefix = 'Subhalo/SubLink/'
    with h5py.File(offsetFile, 'r') as f:    
        N_Galaxy = len(f[prefix+'RowNum'])

    #tree_i = il.sublink.loadTree(basePath,snap_ID,0,fields=fields)
    #print('tree0:',tree_i)
    #tree_i = il.sublink.loadTree(basePath,snap_ID,1,fields=fields)
    #print('tree1:',tree_i)

    #return

    index = -1

    for i in range(N_Galaxy):
        tree_i = il.sublink.loadTree(basePath,snap_ID,i,fields=fields)
        if tree_i is None: #tree_i is None
           continue 
        table_tree_i = pd.DataFrame({'SubhaloID':tree_i['SubhaloID'],
                                     'SnapNum':tree_i['SnapNum'],
                                     'SubfindID':tree_i['SubfindID']})
        if np.mod(i,1000) == 0:
           if i != 0:
              index += 1 
              tree.to_hdf(address+'tree_'+str(index)+'.h5','tree')
              timelist.append(time.time())
              print('Galaxy group index now is:', i)
              print('Generating the '+str(index)+'-th tree table took {0:.1f} seconds.'.format(timelist[-1]-timelist[-2]))            
           
           tree = table_tree_i
        
        else:
           tree = pd.concat([tree, table_tree_i], ignore_index=True, sort=False) 


        if i == N_Galaxy-1:
           index += 1
           tree.to_hdf(address+'tree_'+str(index)+'.h5','tree')
           timelist.append(time.time())
           print('Generating the '+str(index)+'-th tree table took {0:.1f} seconds.'.format(timelist[-1]-timelist[-2])) 


def find_MDB(tree):
    SnapNum_start = tree['SnapNum'][-1]
    #print('all_snapnums',tree['SnapNum'][100:200])
    if len(tree['SnapNum']) <= 100-SnapNum_start:
       return tree 
    index = np.unique(np.flip(tree['SnapNum']), return_index=True)[1][SnapNum_start:100] 
    #print('index is',index)
    for field in tree.keys():
        #print('field is',field)
        if field == 'count':
           tree[field] = 100-SnapNum_start 
        else:
           tree[field] = np.flip(np.flip(tree[field])[index])
 
    #print('tree', tree)
    #check if DescendantID is match to SubhaloID
    for i in range(len(tree['SnapNum'])-1):
        if tree['DescendantID'][i+1] != tree['SubhaloID'][i]:
            print(tree['SnapNum'][i], tree['DescendantID'][i+1], tree['SubhaloID'][i]) 

    return tree


def find_merger_tree(snapnum,ID):

    basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG50-1/output'
    tree = il.sublink.loadTree(basePath, snapnum, ID, fields=['SubhaloID','DescendantID', 'SubfindID', 'SnapNum','SubhaloSFRinRad','SubhaloMassInRadType','SubhaloHalfmassRadType'], onlyMDB=True, onlyMPB=True)
    tree = find_MDB(tree)
    return(tree)
    

#if __name__ == '__main__':
    
   #merger_tree(z=0)
   #test_merger_tree()



