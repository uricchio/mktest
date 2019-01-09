# created by LHU
# 11/28/2016

from sfscoder import command
from scipy.stats import binom
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gamma as gammaFunc
from mpmath import gammainc
from scipy.stats import gamma as gamInt
from scipy.special import polygamma
from scipy.stats import gamma
import subprocess
import sys
import os
import mpmath
import math
import gzip
import numpy as np

class inferTools:

    def __init__(self,Bfile='',al=0.184,be=0.000402,N=500,datadir='../../ABC/output',
                 datadirtemp='../../ABC/output_temp',pref='test',nsims=1000,ABCsumsdir='ABC_sums_neg_test',
                 alfac=1.,befac=1.,testsim = False,reg =True,q=-1,print_als=False,gerp_control=False,Brange=[]):

        self.task_id = 1
        self.testsim = testsim
        self.Bfile = Bfile
        self.al = al
        self.be = be
        self.N = N
        self.datadir = datadir
        self.datadirtemp = datadirtemp
        self.pref = pref
        self.nsims = nsims
        self.ABCsumsdir = ABCsumsdir
        self.alfac = alfac
        self.befac = befac
        self.reg = reg
        self.q = q
        self.print_als = print_als
        self.gerp_control = gerp_control
        self.Brange = Brange

    def alphaF(self,d,d0,p,p0):
        if  d==0 or p0 == 0:
            return 'NA'
        al = 1. - (d0/d)*(p/p0)
        return al

    def get_new_subs(self,num,alfac,befac,B):
        al = self.al
        be = self.be
        N = self.N

        z1 = mpmath.zeta(al,1+be/(2.*B))
        z2 = mpmath.zeta(al,0.5*(2.-1./N +be/B))

        denom = (z1-z2)

        z1n = mpmath.zeta(al*alfac,1+be*befac/(2.*B))
        z2n = mpmath.zeta(al*alfac,0.5*(2.-1./N +be*befac/B))
    
        numerator = (2.**(-al*alfac+al)) * (B**(-al*alfac+al)) * (be**-al)* ((be*befac)**(al*alfac)) * (z1n-z2n)

        return int(round(num*numerator/denom))

    def bias_resample(self,arr,alfac,befac,al,be):
 
        scale0 = gamma.pdf(np.multiply(-1,arr)*self.N*2,a=al,scale=1./be)
        scale1 = gamma.pdf(np.multiply(-1,arr)*self.N*2,a=al*alfac,scale=1./(be*befac))
 
        weight = np.divide(scale1,scale0)
        weightret = np.sum(weight)/len(weight)
        weight = np.divide(weight,np.sum(weight))

        samples  = [int(x) for x in np.random.choice(range(0,len(weight)),len(weight),p=weight)]
        return (samples,weightret)

    def make_data_B(self,B,alfac,befac,wh):
    
        al = self.al
        be = self.be

        #print gamma.pdf(100,a=al,scale=1./be)

        f = os.path.join(self.datadir, 'freq_sel.pi'+B+'.txt')

        fh = open(f,'r')

        fs = []
        sel = []

        ln = 0
        for line in fh:
            if ln % 2 == 0:
                data = [int(x) for x in line.strip().split()]
                fs.append(data) 
 
            else:
                data = [float(x) for x in line.strip().split()]
                sel.append(data)

            ln += 1

        f2 = os.path.join(self.datadir, 'sfs.pi'+B+'.txt')
        f2h = open(f2,'r')

        nums = []

        ln2 = 0
        for line in f2h:
            if ln2 == 0:
                 data = line.strip().split()
             
                 new_subs_4 = self.get_new_subs(int(data[4]),alfac,befac,float(B))
                 new_subs_5 = self.get_new_subs(int(data[5]),alfac,befac,float(B))
                 #print  data[0],data[1],data[2],data[3],new_subs_4,new_subs_5
                 print >> wh, data[0],data[1],data[2],data[3],new_subs_4,new_subs_5
            if ln2 > 0:
                 nums.append(line.strip().split()[1])
            ln2 += 1
            if ln2 == 3:
                break

        # first do neg_strong 
        samples,weight_s = self.bias_resample(sel[1],alfac,befac,al,be)
        freqs = []
  
        for site in samples:
            freqs.append(fs[1][site])

        sfs = np.zeros(2*661-1)
        for fr in freqs:
            sfs[fr]+=1

        print >> wh, B,
        print >> wh, nums[0],
        sfs = np.divide(sfs,np.sum(sfs))
        for val in sfs:
            print >> wh, val,
        print >> wh 
       
        # now do neg_weak
        samples,weight_w = self.bias_resample(sel[0],alfac,befac,al,be)
        freqs = []

        for site in samples:
            freqs.append(fs[0][site])

        sfs = np.zeros(2*661-1)
        for fr in freqs:
            sfs[fr]+=1
        print >> wh, B,
        print >> wh, nums[1],
        sfs = np.divide(sfs,np.sum(sfs))
        for val in sfs:
            print >> wh, val,
        print >> wh 
    
        for line in f2h:
            if ln2 < 3:
                ln2 += 1
                continue
            print >> wh, line.strip()

        f2h.close()
        return (weight_s,weight_w)

    def get_data_B(self,B):

        f = os.path.join(self.datadirtemp,'sfs.'+str(self.task_id)+'.'+self.pref+'.'+str(B)+'.temp.txt')
        fh = open(f, 'r')
   
        dVals = []

        for line in fh:
            data = [float(x) for x in line.strip().split()]
            dVals.extend(data)
            #print dVals
            break 
    
        sfs = {}

        num  = 0
        for line in fh:
            data = [float(x) for x in line.strip().split()]
            sfs[num] = data
            num += 1
        fh.close()
    
        return (dVals, sfs)

    def sample_data(self,tw,ts,Bd,sim_D,sim_s,ws,ww):

        # Bd is [pN, pS, dN, dS]

        tot_poly = Bd[0]+Bd[1]
        tot_fix = Bd[2]+Bd[3]

        choices = range(0,2*661-1)
        sfsN = np.zeros(661*2-1)
        sfsS = np.zeros(661*2-1)
        dN = 0
        dS = 0

        ts_max = 0.00026412640844*0.001
        tw_max = 0.00777990593409*0.001

        # these are the total number of simulated variants in each category

        tot_neg_sim_strong = int(round(sim_s[0][1]*ws))
        tot_syn_sim_strong = sim_s[4][1]
        tot_pos_sim_strong = sim_s[2][1]
        tot_neg_sim_weak = int(round(sim_s[1][1]*ww))
        tot_syn_sim_weak = sim_s[5][1]
        tot_pos_sim_weak = sim_s[3][1]

        dN_w_neg_sim = sim_D[4]
        dN_w_pos_sim = sim_D[2]
        dS_w_sim = sim_D[0]

        dN_s_neg_sim = sim_D[5]
        dN_s_pos_sim = sim_D[3]
        dS_s_sim = sim_D[1]


        # for NS poly 
        # with probability (ts/tsmax)/(ts/tsmax + tw/twmax) select from strong, else weak
    
        s_num = np.random.binomial(tot_poly,(ts/ts_max)/(ts/ts_max+tw/tw_max))
        w_num = tot_poly-s_num

        # with probability Nstrong/(Nstrong+Ndel) get from strong, else del

        s_num_ns = int(round(s_num*(tot_neg_sim_strong+(ts/ts_max)*tot_pos_sim_strong)/(tot_neg_sim_strong+(ts/ts_max)*tot_pos_sim_strong+tot_syn_sim_strong)))
        s_num_syn = s_num-s_num_ns
   
        w_num_ns = int(round(w_num*(tot_neg_sim_weak+(tw/tw_max)*tot_pos_sim_weak)/(tot_neg_sim_weak+(tw/tw_max)*tot_pos_sim_weak+tot_syn_sim_weak)))
        w_num_syn = w_num-w_num_ns

        s_pos = np.random.binomial(s_num_ns,(ts/ts_max)*(tot_pos_sim_strong+0.)/((ts/ts_max)*tot_pos_sim_strong+tot_neg_sim_strong))
        s_neg = s_num_ns-s_pos
    
        w_pos = np.random.binomial(w_num_ns,(tw/tw_max)*(tot_pos_sim_weak+0.)/((tw/tw_max)*tot_pos_sim_weak+tot_neg_sim_weak))
        w_neg = w_num_ns-w_pos

        sfs_NS_loc_s_pos = np.random.choice(choices,int(s_pos),p=sim_s[2][2:])
        sfs_NS_loc_s_neg = np.random.choice(choices,int(s_neg),p=sim_s[0][2:])
    
        for site in sfs_NS_loc_s_pos:
            sfsN[site]+=1
        for site in sfs_NS_loc_s_neg:
            sfsN[site]+=1

        # with probability Nweak/(Nweak+Ndel) get from weak, else del
        sfs_NS_loc_w_pos = np.random.choice(choices,int(w_pos),p=sim_s[3][2:])
        sfs_NS_loc_w_neg = np.random.choice(choices,int(w_neg),p=sim_s[1][2:])

        for site in sfs_NS_loc_w_pos:
            sfsN[site]+=1
        for site in sfs_NS_loc_w_neg:
            sfsN[site]+=1
    
        # for SYN poly
        # with probability (ts/tsmax)/(ts/tsmax + tw/twmax) select from strong, else weak

        sfs_SYN_loc_w = np.random.choice(choices,int(w_num_syn),p=sim_s[5][2:])
        sfs_SYN_loc_s = np.random.choice(choices,int(s_num_syn),p=sim_s[4][2:])

        for site in sfs_SYN_loc_w:
            sfsS[site]+=1
        for site in sfs_SYN_loc_s:
            sfsS[site]+=1

        # For Fixed sites
        # with probability (ts/tsmax)/(ts/tsmax + tw/twmax) select from strong, else weak
        # Sample same total number of sites as in data, with dN/(dN+dS) probability of being NS

        tot_strong_sites = np.random.binomial(tot_fix, (ts/ts_max)/(ts/ts_max+tw/tw_max))
        tot_weak_sites = tot_fix - tot_strong_sites
    
        # dN expected is dN_neg + (t_sim/t_max)*dN_pos
        # dS unaltered

        dN_w_exp = sim_D[4]+int(round((tw/tw_max)*sim_D[2]))
        dN_w = np.random.binomial(tot_weak_sites,dN_w_exp/(dN_w_exp+sim_D[0]+0.))
 
        dN_w_plus = np.random.binomial(dN_w,int(round((tw/tw_max)*sim_D[2]))/(dN_w_exp+0.))

        dN_s_exp = sim_D[5]+int(round((ts/ts_max)*sim_D[3]))
        dN_s = np.random.binomial(tot_strong_sites,dN_s_exp/(dN_s_exp+sim_D[1]+0.))

        dN_s_plus = np.random.binomial(dN_s,int(round((ts/ts_max)*sim_D[3]))/(dN_s_exp+0.))
        
        dS_s = tot_strong_sites - dN_s
        dS_w = tot_weak_sites - dN_w 

        dN = dN_w+dN_s
        dS = dS_w+dS_s

        al = 0
        al_w = 0
        al_s = 0

        if dN != 0:
            al = (dN_w_plus+dN_s_plus+0.)/dN
            al_w = (dN_w_plus+0.)/dN
            al_s = (dN_s_plus+0.)/dN


        return (sfsN, sfsS, dN_s, dN_w, dS, al,al_s,al_w) 

    def simulate(self):  
        # inputs & params

        Bfile  = self.Bfile
        nsims = self.nsims
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            self.task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

        # now load all the B values for polymorphic sites, fixed sites
        Bh = open(Bfile, 'r')

        # Bdata is pN, pS, dN, dS
        Bdata = {}
        for i in range(4,21):
            Bdata[i] = np.zeros(4)

        for line in Bh:
            data = line.strip().split() 
            if float(data[-1])/1000. < 0.175:
                continue
            i = int(np.ceil(20.*float(data[-1])/1000.))
            Bdata[i][0] += int(data[1])
            Bdata[i][1] += int(data[3])
            Bdata[i][2] += int(data[5])
            Bdata[i][3] += int(data[6])

        # Loop: first sample theta_weak, theta_strong
        # For each B value in B_values, sample a polymorphic site with same B. 
        # If site is NS, sample from NS frequency spectrum. 

        sum_stats = np.add([1,2,5,10,20,50,100,200,500,1000],-1)

        of = os.path.join(self.ABCsumsdir,'sum.'+str(self.task_id)+'.'+self.pref+'.txt')
        ofh = open(of,'w')

        Bvals = ['0.2','0.25','0.3','0.35','0.4','0.45','0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','0.999']
        if self.testsim:
            Bvals = [Bvals[-1]]

        sfs_NS = np.zeros(2*661-1)
        sfs_SYN = np.zeros(2*661-1)
        dN = 0
        dS = 0
        dN_w = 0
        dN_s = 0
        for nsim in range(nsims):
            if not self.testsim:
                sfs_NS = np.zeros(2*661-1)
                sfs_SYN = np.zeros(2*661-1)
                dN = 0
                dS = 0
                dN_w = 0
                dN_s = 0
    
            alfac = 1
            befac = 1
            if self.testsim:
                alfac = self.alfac 
                befac = self.befac 
            else:
                alfac = 2**np.random.uniform(-2,2) 
                befac = 2**np.random.uniform(-2,2)
 
            weights_s = []
            weights_w = []

            #first make sim data
            for B in Bvals:
                wh = open(os.path.join(self.datadirtemp,'sfs.'+str(self.task_id)+'.'+self.pref+'.'+str(B)+'.temp.txt'),'w')
                ws,ww = self.make_data_B(B,alfac,befac,wh) 
                weights_s.append(ws)
                weights_w.append(ww)
                wh.close()

            # next load all the simulation data
            sim_sfs = {}
            sim_D = {}
            
            if not self.Brange:
                if not self.testsim: 
                    (sim_D[4], sim_sfs[4]) = self.get_data_B(0.2)
                    (sim_D[5], sim_sfs[5])  = self.get_data_B(0.25)
                    (sim_D[6], sim_sfs[6]) = self.get_data_B(0.3)
                    (sim_D[7], sim_sfs[7]) = self.get_data_B(0.35)
                    (sim_D[8], sim_sfs[8]) = self.get_data_B(0.4)
                    (sim_D[9], sim_sfs[9]) = self.get_data_B(0.45)
                    (sim_D[10], sim_sfs[10]) = self.get_data_B(0.5)
                    (sim_D[11], sim_sfs[11]) = self.get_data_B(0.55)
                    (sim_D[12], sim_sfs[12]) = self.get_data_B(0.6)
                    (sim_D[13], sim_sfs[13]) = self.get_data_B(0.65)
                    (sim_D[14], sim_sfs[14]) = self.get_data_B(0.7)
                    (sim_D[15], sim_sfs[15]) = self.get_data_B(0.75)
                    (sim_D[16], sim_sfs[16]) = self.get_data_B(0.8)
                    (sim_D[17], sim_sfs[17])  = self.get_data_B(0.85)
                    if not self.gerp_control:
                        (sim_D[18], sim_sfs[18]) = self.get_data_B(0.9)
                        (sim_D[19], sim_sfs[19]) = self.get_data_B(0.95)
                if not self.gerp_control: 
                     (sim_D[20], sim_sfs[20]) = self.get_data_B(0.999)
            else:
                for B in range(self.Brange[0],self.Brange[1]+1):
                    if (float(B)/20.) < 1.:
                        (sim_D[B], sim_sfs[B]) = self.get_data_B(float(B)/20.)
                    else:        
                        (sim_D[B], sim_sfs[B]) = self.get_data_B(0.999)

            for B in Bvals: 
                os.remove(os.path.join(self.datadirtemp,'sfs.'+str(self.task_id)+'.'+self.pref+'.'+str(B)+'.temp.txt'))

            theta_strong = np.random.random()*0.000264126408446*0.001
            theta_weak =  np.random.random()*0.00777990593409*0.001
            
            if self.testsim:
                theta_strong = 0.000264126408446*0.001
                theta_weak =  0.00777990593409*0.001

            weighted_al =[]
            weighted_al_s = []
            weighted_al_w = []
            al_noBGS = 0

            myran = range(4,21)
            if self.testsim:
                myran = range(20,21) 
            if self.gerp_control: 
                myran = range(4,18)
            if self.Brange:
                myran = range(self.Brange[0],self.Brange[1]+1) 


            for i in myran:
                (sfs_NS_l, sfs_SYN_l, dN_s_l, dN_w_l, dS_l, al_loc, al_s, al_w)  = self.sample_data(theta_weak,theta_strong,Bdata[i],sim_D[i],sim_sfs[i],weights_s[i-myran[0]],weights_w[i-myran[0]])
    
                sfs_NS = np.add(sfs_NS, sfs_NS_l)
                sfs_SYN = np.add(sfs_SYN, sfs_SYN_l)
                dN += dN_s_l+dN_w_l
                dN_s += dN_s_l
                dN_w += dN_w_l
                dS += dS_l
                weighted_al.append([dN_s_l+dN_w_l,al_loc])
                weighted_al_w.append([dN_w_l+dN_s_l,al_w])
                weighted_al_s.append([dN_s_l+dN_w_l,al_s])
                if i == 20:
                    al_noBGS =  al_loc
                
            av = 0
            tot = 0
            for thing in weighted_al:
                av +=  thing[1]*thing[0]
                tot += thing[0]
            av /= tot
    
            av_w = 0
            tot_w = 0
            for thing in weighted_al_w:
                av_w+=  thing[1]*thing[0]
                tot_w += thing[0]
            av_w /= tot_w
    
            av_s = 0
            tot_s = 0
            for thing in weighted_al_s:
                av_s+=  thing[1]*thing[0]
                tot_s += thing[0]
            av_s /= tot_s
   
            if not self.testsim:
                l = range(0,len(sfs_SYN[:-1]))
                l.reverse()
                for i in l:
                    sfs_SYN[i] += sfs_SYN[i+1]
                    sfs_NS[i] += sfs_NS[i+1] 
                ofh.write(str(av_s)+' '+str(av_w)+' '+str(av)+' '+str(alfac)+' '+str(befac)+' '+str(al_noBGS)+' '+str(theta_weak/(0.00777990593409*0.001))+' '+str(theta_strong/(0.000264126408446*0.001))+' ')
                for i in sum_stats:
                    ofh.write(str(self.alphaF(dN+0.,dS+0.,sfs_NS[i]+0.,sfs_SYN[i]+0.))+' ')
                ofh.write('\n')


        if self.testsim:
            l = range(0,len(sfs_SYN[:-1]))
            ofh.write('# '+str(dN)+' '+str(dS)+'\n')
            for i in l:
                ofh.write(str(sfs_SYN[i])+' '+str(sfs_NS[i])+'\n')       
            ofh.close()
    

    def simulate_resamp(self,inpfile):  
        # inputs & params

        iph = gzip.open(inpfile,'r')
        params = []
        for line in iph:
            fields = line.strip().split()
            params.append([float(x) for x in [fields[3],fields[4],fields[6],fields[7]]])

        Bfile  = self.Bfile
        nsims = self.nsims
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            self.task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

        # now load all the B values for polymorphic sites, fixed sites
        Bh = open(Bfile, 'r')

        # Bdata is pN, pS, dN, dS
        Bdata = {}
        for i in range(4,21):
            Bdata[i] = np.zeros(4)

        for line in Bh:
            data = line.strip().split() 
            if float(data[-1])/1000. < 0.175:
                continue
            i = int(np.ceil(20.*float(data[-1])/1000.))
            Bdata[i][0] += int(data[1])
            Bdata[i][1] += int(data[3])
            Bdata[i][2] += int(data[5])
            Bdata[i][3] += int(data[6])

        # Interestingly dN/dS is fairly strongly correlated with B
        #for i in sorted(Bdata):
        #    print i, (Bdata[i][2]+0.)/Bdata[i][3]

        # Loop: first sample theta_weak, theta_strong
        # For each B value in B_values, sample a polymorphic site with same B. 
        # If site is NS, sample from NS frequency spectrum.  With prob theta_plus/

        #sum_stats = np.add([1,2,5,10,20,50,100,200,500,1000],-1)
        sum_stats = range(0,661*2-1)

        of = os.path.join(self.ABCsumsdir,'sum.'+str(self.task_id)+'.'+self.pref+'.txt')
        ofh = open(of,'w')

        Bvals = ['0.2','0.25','0.3','0.35','0.4','0.45','0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','0.999']
        if self.testsim:
            Bvals = [Bvals[-1]]

        sfs_NS = np.zeros(2*661-1)
        sfs_SYN = np.zeros(2*661-1)
        dN = 0
        dS = 0
        dN_w = 0
        dN_s = 0
        for nsim in range(nsims):
            if not self.testsim:
                sfs_NS = np.zeros(2*661-1)
                sfs_SYN = np.zeros(2*661-1)
                dN = 0
                dS = 0
                dN_w = 0
                dN_s = 0
    
            alfac = 1
            befac = 1
            if self.testsim:
                alfac = self.alfac 
                befac = self.befac 
            else:
                alfac = params[nsim][0]
                befac = params[nsim][1]
 
            weights_s = []
            weights_w = []

            #first make sim data
            for B in Bvals:
                wh = open(os.path.join(self.datadirtemp,'sfs.'+str(self.task_id)+'.'+self.pref+'.'+str(B)+'.temp.txt'),'w')
                ws,ww = self.make_data_B(B,alfac,befac,wh) 
                weights_s.append(ws)
                weights_w.append(ww)
                wh.close()

            # next load all the simulation data
            sim_sfs = {}
            sim_D = {}
            

            if self.reg:
                if not self.testsim: 
                    (sim_D[4], sim_sfs[4]) = self.get_data_B(0.2)
                    (sim_D[5], sim_sfs[5])  = self.get_data_B(0.25)
                    (sim_D[6], sim_sfs[6]) = self.get_data_B(0.3)
                    (sim_D[7], sim_sfs[7]) = self.get_data_B(0.35)
                    (sim_D[8], sim_sfs[8]) = self.get_data_B(0.4)
                    (sim_D[9], sim_sfs[9]) = self.get_data_B(0.45)
                    (sim_D[10], sim_sfs[10]) = self.get_data_B(0.5)
                    (sim_D[11], sim_sfs[11]) = self.get_data_B(0.55)
                    (sim_D[12], sim_sfs[12]) = self.get_data_B(0.6)
                    (sim_D[13], sim_sfs[13]) = self.get_data_B(0.65)
                    (sim_D[14], sim_sfs[14]) = self.get_data_B(0.7)
                    (sim_D[15], sim_sfs[15]) = self.get_data_B(0.75)
                    (sim_D[16], sim_sfs[16]) = self.get_data_B(0.8)
                    (sim_D[17], sim_sfs[17])  = self.get_data_B(0.85)
                    (sim_D[18], sim_sfs[18]) = self.get_data_B(0.9)
                    (sim_D[19], sim_sfs[19]) = self.get_data_B(0.95)
                (sim_D[20], sim_sfs[20]) = self.get_data_B(0.999)

            else:
                if self.q == 1:  
                    (sim_D[4], sim_sfs[4]) = self.get_data_B(0.2)
                    (sim_D[5], sim_sfs[5])  = self.get_data_B(0.25)
                    (sim_D[6], sim_sfs[6]) = self.get_data_B(0.3)
                    #(sim_D[7], sim_sfs[7]) = self.get_data_B(0.35)
                    #(sim_D[8], sim_sfs[8]) = self.get_data_B(0.4)
                    #(sim_D[9], sim_sfs[9]) = self.get_data_B(0.45)
                    #(sim_D[10], sim_sfs[10]) = self.get_data_B(0.5)
                    #(sim_D[11], sim_sfs[11]) = self.get_data_B(0.55)
                    #(sim_D[12], sim_sfs[12]) = self.get_data_B(0.6)
                if self.q == 2:
                    #(sim_D[10], sim_sfs[10]) = self.get_data_B(0.5)
                    (sim_D[12], sim_sfs[12]) = self.get_data_B(0.6)
                    (sim_D[13], sim_sfs[13]) = self.get_data_B(0.65)
                    (sim_D[14], sim_sfs[14]) = self.get_data_B(0.7)
                    #(sim_D[15], sim_sfs[15]) = self.get_data_B(0.75)
                    #(sim_D[16], sim_sfs[16]) = self.get_data_B(0.8)
                if self.q == 3:
                    #(sim_D[17], sim_sfs[17])  = self.get_data_B(0.85)
                    (sim_D[18], sim_sfs[18]) = self.get_data_B(0.9)
                    (sim_D[19], sim_sfs[19]) = self.get_data_B(0.95)
                    (sim_D[20], sim_sfs[20]) = self.get_data_B(0.999)

            for B in Bvals: 
                os.remove(os.path.join(self.datadirtemp,'sfs.'+str(self.task_id)+'.'+self.pref+'.'+str(B)+'.temp.txt'))

            #theta_strong = np.random.random()*0.000264126408446*0.001
            #theta_weak =  np.random.random()*0.00777990593409*0.001
            theta_strong = params[nsim][3]*0.000264126408446*0.001
            theta_weak = params[nsim][2]*0.00777990593409*0.001


            if self.testsim:
                theta_strong = 0.000264126408446*0.001
                theta_weak =  0.00777990593409*0.001

            weighted_al =[]
            weighted_al_s = []
            weighted_al_w = []
            al_noBGS = 0

            if self.reg:
                myran = range(4,21)
                if self.testsim:
                    myran = range(20,21) 
            else:
                myran = range(4,7)
                if self.q == 1:
                    myran = range(4,7) 
                if self.q == 2:
                    myran = range(12,15) 
                if self.q == 3:
                    myran = range(18,21) 

            all_als = []

            for i in myran:
                (sfs_NS_l, sfs_SYN_l, dN_s_l, dN_w_l, dS_l, al_loc, al_s, al_w)  = self.sample_data(theta_weak,theta_strong,Bdata[i],sim_D[i],sim_sfs[i],weights_s[i-myran[0]],weights_w[i-myran[0]])
    
                all_als.append(al_loc)
                sfs_NS = np.add(sfs_NS, sfs_NS_l)
                sfs_SYN = np.add(sfs_SYN, sfs_SYN_l)
                dN += dN_s_l+dN_w_l
                dN_s += dN_s_l
                dN_w += dN_w_l
                dS += dS_l
                weighted_al.append([dN_s_l+dN_w_l,al_loc])
                weighted_al_w.append([dN_w_l+dN_s_l,al_w])
                weighted_al_s.append([dN_s_l+dN_w_l,al_s])
                if i == 20:
                    al_noBGS =  al_loc
                
            av = 0
            tot = 0
            for thing in weighted_al:
                av +=  thing[1]*thing[0]
                tot += thing[0]
            av /= tot
    
            av_w = 0
            tot_w = 0
            for thing in weighted_al_w:
                av_w+=  thing[1]*thing[0]
                tot_w += thing[0]
            av_w /= tot_w
    
            av_s = 0
            tot_s = 0
            for thing in weighted_al_s:
                av_s+=  thing[1]*thing[0]
                tot_s += thing[0]
            av_s /= tot_s
   
            if self.print_als and not self.testsim:
                for i in range(len(all_als)):
                    ofh.write(str(all_als[i])+' ')
                ofh.write('\n')

            if not self.testsim and not self.print_als:
                l = range(0,len(sfs_SYN[:-1]))
                l.reverse()
                for i in l:
                    sfs_SYN[i] += sfs_SYN[i+1]
                    sfs_NS[i] += sfs_NS[i+1] 
                ofh.write(str(av_s)+' '+str(av_w)+' '+str(av)+' '+str(alfac)+' '+str(befac)+' '+str(al_noBGS)+' '+str(theta_weak/(0.00777990593409*0.001))+' '+str(theta_strong/(0.000264126408446*0.001))+' ')
                for i in sum_stats:
                    ofh.write(str(self.alphaF(dN+0.,dS+0.,sfs_NS[i]+0.,sfs_SYN[i]+0.))+' ')
                ofh.write('\n')

        if self.testsim:
            l = range(0,len(sfs_SYN[:-1]))
            ofh.write('# '+str(dN)+' '+str(dS)+'\n')
            for i in l:
                ofh.write(str(sfs_SYN[i])+' '+str(sfs_NS[i])+'\n')       
            ofh.close()
