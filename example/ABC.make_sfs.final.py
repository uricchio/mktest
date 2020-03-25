#!/usr/bin/python
#$ -e sim.div.log
#$ -o sim.div.log
#$ -S /usr/bin/python
#$ -cwd
#$ -r yes
#$ -l h_rt=48:00:00
#$ -t 1-1

from sfscoder import sfs
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np

fold = sys.argv[1]
B = sys.argv[2]

files = [join(fold,f) for f in listdir(fold) if isfile(join(fold, f))]

dS_W = 0
dS_S = 0
dN_W = 0
dN_S = 0
dN_negW = 0
dN_negS = 0

sfs_pos_weak = np.zeros(2*661-1)
sfs_neg_weak = np.zeros(2*661-1)
sfs_pos_strong = np.zeros(2*661-1)
sfs_neg_strong = np.zeros(2*661-1)
sfs_syn_weak = np.zeros(2*661-1)
sfs_syn_strong = np.zeros(2*661-1)

freq_sel_weak = []
freq_sel_strong = []

itnum = 0
for file in files:
    data = sfs.SFSData(file = file)
    data.get_sims()
    for sim in data.sims:
        for mut in sim.muts:
            if mut.locus != 1 and mut.locus != 2:
                continue  
            if mut.fixed_pop[0] and mut.fit == 0.:
                if mut.locus == 1:
                    dS_W += 1.
                if mut.locus == 2:
                    dS_S += 1.
            if mut.fixed_pop[0] and mut.fit != 0.:
                if mut.locus == 1:
                    if mut.fit < 0.:
                        dN_negW += 1
                    else:
                        dN_W += 1
                if mut.locus == 2:
                    if mut.fit < 0.:
                        dN_negS += 1
                    else:
                        dN_S += 1
            if not mut.fixed_pop[0] and mut.fit == 0.:
                if mut.locus == 1:
                    sfs_syn_weak[len(mut.chrs[0])-1] += 1
                if mut.locus == 2: 
                    sfs_syn_strong[len(mut.chrs[0])-1] += 1
            if not mut.fixed_pop[0] and mut.fit > 0.:
                if mut.locus == 1:
                   sfs_pos_weak[len(mut.chrs[0])-1] += 1
                if mut.locus == 2:
                   sfs_pos_strong[len(mut.chrs[0])-1] += 1
            if not mut.fixed_pop[0] and mut.fit < 0.:
                if mut.locus == 1:
                   sfs_neg_weak[len(mut.chrs[0])-1] += 1
                   freq_sel_weak.append([len(mut.chrs[0])-1,mut.fit])

                if mut.locus == 2:
                   sfs_neg_strong[len(mut.chrs[0])-1] += 1
                   freq_sel_strong.append([len(mut.chrs[0])-1,mut.fit])
        itnum += 1

def print_vec(B,vec,ofh):
    ofh.write(str(B) +' '+str(np.sum(vec))+' ')
    vec = np.multiply(vec,1./np.sum(vec))
    for site in vec:
        ofh.write( str(site)+' ')
    ofh.write('\n')

of = os.path.join(os.getcwd(),'output4','sfs.pi'+B+'.txt')
of2 = os.path.join(os.getcwd(),'output4','freq_sel.pi'+B+'.txt')
ofh = open(of,'w')
ofh2 = open(of2,'w')

ofh.write(str(dS_W)+' '+str(dS_S)+' '+str(dN_W)+' '+str(dN_S)+' '+str(dN_negW)+' '+str(dN_negS)+'\n') 

print_vec(B,sfs_neg_strong,ofh)
print_vec(B,sfs_neg_weak,ofh)
print_vec(B,sfs_pos_strong,ofh)
print_vec(B,sfs_pos_weak,ofh)
print_vec(B,sfs_syn_strong,ofh)
print_vec(B,sfs_syn_weak,ofh)

for thing in freq_sel_weak:
    print >> ofh2, thing[0],
print >> ofh2

for thing in freq_sel_weak:
    print >> ofh2, thing[1],
print >> ofh2

for thing in freq_sel_strong:
    print >> ofh2, thing[0],
print >> ofh2

for thing in freq_sel_strong:
    print >> ofh2, thing[1],
print >> ofh2

ofh2.close()
ofh.close()

