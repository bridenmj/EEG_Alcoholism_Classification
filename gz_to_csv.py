# coding: utf-8

# In[88]:


import os
import gzip
import sys
import shutil
import subprocess
import codecs
#convert .rd files to 
# set base_dir to the unzipped eeg file that you wish to convert to .csv

#the path to the full_eeg dataset converted from .zip
base_dir = '/Users/mike/Documents/ucsc/cmps240/EEG_classification/EEG_Alcoholism_Classification/Data/SMNI_CMI_TRAIN'
directory_dir =''

#create S1,S2,S3 directories
base_dir1=os.path.join(base_dir, "S1")
base_dir2=os.path.join(base_dir, "S2")
base_dir3=os.path.join(base_dir, "S3")
os.mkdir(base_dir1)
os.mkdir(base_dir2)
os.mkdir(base_dir3)
os.mkdir(base_dir1+"/alcoholic")
os.mkdir(base_dir1+"/control")
os.mkdir(base_dir2+"/alcoholic")
os.mkdir(base_dir2+"/control")
os.mkdir(base_dir3+"/alcoholic")
os.mkdir(base_dir3+"/control")

print("extracting...")
#iterate over files in base directory
for directory in os.listdir(base_dir):
    if directory =='.DS_Store' or directory =='README':
            continue
    directory_dir = os.path.join(base_dir, directory)
    
    #iterate bottom level files
    for filename in os.listdir(directory_dir):
        fname = os.path.join(directory_dir, filename)
        if fname[-3:] == ".gz":
            #extract
            subprocess.call(['gunzip -q ' + fname], shell = True)
            
 
#convert to csv
for directory in os.listdir(base_dir):
    print("converting to .csv...") 
    if directory =='.DS_Store' or directory =='README':
        continue
        
    #update directory to file
    directory_dir = os.path.join(base_dir, directory)
    #create new_filename variable
    new_filename=""
    
    #iterate bottom level files
    for filename in os.listdir(directory_dir):
        if filename =='.DS_Store' or 'alcoholic'== filename or filename == 'control':
            continue
        fname = os.path.join(directory_dir, filename)
        f = open(fname, "r")
        lines = [] #Declare an empty list named "lines"
        #open files
        with open(fname, "r") as f: 
            #store file lines in lines
            for line in f:
                
                lines.append(line)
        f.close()
    
        stringvar1 = lines[0]
        stringvar2 = lines[3]
        stringvar3 = filename[-3:]
        
        #construct file name
        if stringvar2[2:4] == "S1":
            new_filename=stringvar1[5]+"_"+stringvar2[2:4]+"_"+stringvar1[10:13]+"_"+stringvar3+".csv"
        elif 'nomatch' in stringvar2 :
            new_filename=stringvar1[5]+"_"+"S3"+"_"+stringvar1[10:13]+"_"+stringvar3+".csv"
        elif 'match' in stringvar2 and "non" not in stringvar2:
             new_filename=stringvar1[5]+"_"+"S2"+"_"+stringvar1[10:13]+"_"+stringvar3+".csv"
        
        #create path to correct class subdirectoy
        if new_filename[2:4] == "S1":
            if new_filename[0] == "a":
                tmp = os.path.join(base_dir1, 'alcoholic')
                new_filename = os.path.join(tmp, new_filename)
            elif new_filename[0] == "c":
                tmp = os.path.join(base_dir1, 'control')
                new_filename = os.path.join(tmp, new_filename)
        elif new_filename[2:4] == "S2":
            if new_filename[0] == "a":
                tmp = os.path.join(base_dir2, 'alcoholic')
                new_filename = os.path.join(tmp, new_filename)
            elif new_filename[0] == "c":
                tmp = os.path.join(base_dir2, 'control')
                new_filename = os.path.join(tmp, new_filename)
        elif new_filename[2:4] == "S3":
            if new_filename[0] == "a":
                tmp = os.path.join(base_dir3, 'alcoholic')
                new_filename = os.path.join(tmp, new_filename)
            elif new_filename[0] == "c":
                tmp = os.path.join(base_dir3, 'control')
                new_filename = os.path.join(tmp, new_filename)
        
        #write to file
        new_file = open(new_filename,"w+")
        new_file.write("trial channel time voltage\n")
        lines = lines[4:]
        for line in lines:
            stringvar = line
            #remove title write observation to new_file
            if stringvar[0] != "#":
                x1 = stringvar.split(" ")
                new_file.write(x1[1]+" "+x1[2]+" "+x1[3])
        new_file.close()
print('done')           
                
                
                
                
                
                
                
                


                
                
                
                
                
                

