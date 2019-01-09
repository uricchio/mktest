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
import subprocess
import sys
import os
import mpmath
import math
import numpy as np

class AsympMK:

    def __init__(self,gam_neg=-40,N=250,theta_f=1e-5,theta_mid_neutral=1e-3,
                 alLow=0.2,alTot=0.2,gL=10,gH=200,al=0.184,be=0.000402,B=1.,
                 pposL=0.001,pposH=0.0,n=10,Lf=10**6,rho=0.001,neut_mid=False,
                 L_mid=150,pref="",nsim=100,TE=5.,demog=False,ABC=False,al2=0.0415,
                 be2=0.00515625,gF=False,skip=0,expan=1.98,scratch=False):

         self.skip = skip
         self.gam_neg = gam_neg
         self.N = N
         self.NN = 2*N
         self.theta_f = theta_f
         self.theta_mid_neutral = theta_mid_neutral
         self.alLow = alLow
         self.alTot = alTot
         self.gL = gL
         self.gH = gH
         self.al = al
         self.be = be
         self.al2 = al2
         self.be2 = be2
         self.B = B
         self.task_id = 1
         self.pposL = pposL
         self.pposH = pposH
         self.n = n
         self.nn = 2*n
         self.alpha_x = np.zeros(self.nn-1)
         self.Lf = Lf 
         self.L_mid = L_mid
         self.pref = pref
         self.nsim=nsim
         self.gF = gF
         self.cumuSfsNeut= []
         self.cumuSfsSel = []
         self.TE = TE
         self.demog = demog
         self.ABC=ABC
         self.expan=expan
         self.scratch=scratch
         while self.L_mid % 3 != 0:
             self.L_mid +=1 
         self.rho = rho
         self.neut_mid = neut_mid
         if 'SGE_TASK_ID' in os.environ:
             self.task_id = int(os.environ['SGE_TASK_ID'])
         if 'SLURM_ARRAY_TASK_ID' in os.environ:
             self.task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
         self.task_id = self.task_id + skip

    def GamSfsNeg(self, x):
        beta = self.be/(1.*self.B)
        return (2.**-self.al)*(beta**self.al)*(-mpmath.zeta(self.al,x+beta/2.) + mpmath.zeta(self.al,(2+beta)/2.))/((-1.+x)*x)

    def SfsPos(self, gamma, x):
        gam = gamma*self.B
        return 0.5*(mpmath.exp(2*gam)*(1-mpmath.exp(-2.*gam*(1.-x)))/((mpmath.exp(2*gam)-1.)*x*(1.-x)))

    def FullSfs(self, gamma, ppos, x):
        if x > 0 and x < 1.:
            #S = abs(self.gam_neg/(1.*self.NN))
            #r = self.rho/(2.*self.NN)
            #u = self.theta_f/(2.*self.NN)
            #s = gamma/(self.NN*1.)
            #p0 = polygamma(1,(s+S)/r)
            #p1 = polygamma(1,1.+(r*self.Lf+s+S)/r)
            #pposplus = ppos*mpmath.exp(-2.*S*u*(p0-p1)/r**2)
            return ppos*self.SfsPos(gamma, x) + (1.-ppos)*self.GamSfsNeg(x)
        return 0.
 
    def FullPos(self, gamma, ppos, x):
        if x > 0 and x < 1.:
            #S = abs(self.gam_neg/(1.*self.NN))
            #r = self.rho/(2.*self.NN)
            #u = self.theta_f/(2.*self.NN)
            #s = gamma/(self.NN*1.)
            #p0 = polygamma(1,(s+S)/r)
            #p1 = polygamma(1,1.+(r*self.Lf+s+S)/r)
            #pposplus = ppos*mpmath.exp(-2.*S*u*(p0-p1)/r**2)
            return ppos*self.SfsPos(gamma, x)
        return 0.
 
    def FullNeg(self, ppos, x):
        if x > 0 and x < 1.:
            return (1.-ppos)*self.GamSfsNeg(x)
        return 0.

    def binomOp(self):
        NN = int(round(self.NN*self.B))
        samps = [[i for j in range(0,NN+1)] for i in range(0,self.nn+1)]
        sampFreqs = [[j/(NN+0.) for j in range(0,NN+1)] for i in range(0,self.nn+1)]
        return binom.pmf(samps,self.nn,sampFreqs)

    def DiscSFSSel(self,gamma,ppos):
        NN = int(round(self.NN*self.B))
        dFunc = np.vectorize(self.FullSfs)
        return np.multiply(1./(NN+0.),dFunc(gamma,ppos,[i/(NN+0.) for i in range(0,NN+1)]))
    
    def DiscSFSSelPos(self,gamma,ppos):
        NN = int(round(self.NN*self.B))
        dFunc = np.vectorize(self.FullPos)
        return np.multiply(1./(NN+0.),dFunc(gamma,ppos,[i/(NN+0.) for i in range(0,NN+1)]))
    
    def DiscSFSSelNeg(self,ppos):
        NN = int(round(self.NN*self.B))
        dFunc = np.vectorize(self.FullNeg)
        return np.multiply(1./(NN+0.),dFunc(ppos,[i/(NN+0.) for i in range(0,NN+1)]))

    def DiscSFSNeut(self):
        NN = int(round(self.NN*self.B))
        def sfs(i):
            if i > 0 and i < NN:
                 return 1./(i+0.)
            return 0.
        sfs = np.vectorize(sfs)
        return sfs([(i+0.) for i in xrange(0,NN+1)])

    def DiscSFSSelDown(self, gamma, ppos):
        return self.DiscSFSSelPosDown(gamma,ppos)+self.DiscSFSSelNegDown(ppos)
    
    def DiscSFSSelPosDown(self, gamma, ppos):
        S = abs(self.gam_neg/(1.*self.NN))
        r = self.rho/(2.*self.NN)
        u = self.theta_f/(2.*self.NN)
        s = gamma/(self.NN*1.)
        p0 = polygamma(1,(s+S)/r)
        p1 = polygamma(1,1.+(r*self.Lf+s+S)/r)
        red_plus = mpmath.exp(-2.*S*u*(p0-p1)/r**2)
        return (self.theta_mid_neutral)*red_plus*0.745*(np.dot(self.binomOp(),self.DiscSFSSelPos(gamma,ppos)))[1:-1]
    
    def DiscSFSSelNegDown(self, ppos):
        return self.B*(self.theta_mid_neutral)*0.745*(np.dot(self.binomOp(),self.DiscSFSSelNeg(ppos)))[1:-1]

    def DiscSFSNeutDown(self):
        return self.B*(self.theta_mid_neutral)*0.255*(np.dot(self.binomOp(),self.DiscSFSNeut()))[1:-1]

    def fixNeut(self):
        return 0.255*(1./(self.B*self.NN))

    #def fixNeg(self,ppos):
    #    return 0.745*(1-ppos)*(2**(-self.al))*(self.be**self.al)*(-mpmath.zeta(self.al,(2.+self.be)/2.)+mpmath.zeta(self.al,0.5*(2-1./self.NN+self.be)))
    #  0.745*(1-ppos)*(2**(-alpha))*(B**-alpha)*(beta**alpha)*(-mpmath.zeta(alpha,1.+beta/(2.*B))+mpmath.zeta(alpha,0.5*(2-1./NN+beta/B)))

    def fixNegB(self,ppos):
        return 0.745*(1-ppos)*(2**(-self.al))*(self.B**(-self.al))*(self.be**self.al)*(-mpmath.zeta(self.al,1.+self.be/(2.*self.B))+mpmath.zeta(self.al,0.5*(2-1./(self.N*self.B)+self.be/self.B)))

    def pFix(self,gamma):
        s = gamma/(self.NN+0.)
        pfix = (1.-mpmath.exp(-2.*s))/(1.-mpmath.exp(-2.*gamma))
        if s >= 0.1:
            pfix = mpmath.exp(-(1.+s))
            lim = 0
            while(lim < 200):
                pfix = mpmath.exp((1.+s)*(pfix-1.))
                lim +=1
            pfix = 1-pfix
        return pfix
        
    def fixPosSim(self,gamma,ppos):
        # large effect mutations Barton 1995 (eqn. 22a)  
        # Old version: return (1./(self.B))*0.745*ppos*(1-mpmath.exp(-gamma/(self.NN+0.)))/(1-mpmath.exp(-2.*gamma))
        # small effect mutations Barton 1995 (eqn. 22b)
        # Old version: return 0.745*ppos*(1-mpmath.exp(-gamma/(self.NN+0.)))/(1-mpmath.exp(-2.*gamma))
        #U = self.theta_f*self.Lf/(2.*self.NN)
        #R = 2*self.Lf*self.rho/(2.*self.NN)
        #TH = abs((gamma+0.)/self.gam_neg)
        #denom = 1./(1+4.*U*(-1.*self.gam_neg/(self.NN+0.)))
        #if TH >= 1.:
        #    return 0.745*ppos*mpmath.exp((-2.*U/(R*TH))*(1-1./(3.*TH)))*self.pFix(gamma)*denom
        #else:
            #print mpmath.exp((-2.*U/R)*(1.-TH/3.))*denom
        #    return 0.745*ppos*mpmath.exp((-2.*U/R)*(1.-TH/3.))*self.pFix(gamma)*denom
        
        S = abs(self.gam_neg/(1.*self.NN))
        r = self.rho/(2.*self.NN)
        u = self.theta_f/(2.*self.NN)
        s = gamma/(self.NN*1.)
        p0 = polygamma(1,(s+S)/r)
        p1 = polygamma(1,(r+self.Lf*r+s+S)/r)
        #CC = 2*s/self.pFix(gamma)
        CC = 1.
        #print mpmath.exp(-2.*S*u*(p0-p1)/r**2)
        return 0.745*ppos*mpmath.exp(-2.*S*u*(p0-p1)*CC**2/r**2)*self.pFix(gamma)

    def alphaExpSimLow(self,pposL,pposH):
        return self.fixPosSim(self.gL,0.5*pposL)/(self.fixPosSim(self.gL,0.5*pposL)+self.fixPosSim(self.gH,0.5*pposH)+self.fixNegB(0.5*pposL+0.5*pposH))

    def alphaExpSimTot(self,pposL,pposH):
        return (self.fixPosSim(self.gL,0.5*pposL)+self.fixPosSim(self.gH,0.5*pposH))/(self.fixPosSim(self.gL,0.5*pposL)+self.fixPosSim(self.gH,0.5*pposH)+self.fixNegB(0.5*pposL+0.5*pposH))

    def solvEqns(self,params):
        pposL,pposH = params
        return (self.alphaExpSimTot(pposL,pposH)-self.alTot,self.alphaExpSimLow(pposL,pposH)-self.alLow)

    def setPpos(self):
        pposL,pposH =  fsolve(self.solvEqns,(0.001,0.001))
        #print self.alphaExpSimLow(pposL,pposH)
        #print self.alphaExpSimTot(pposL,pposH)
        if pposL < 0.:
            pposL = 0.
        if pposH < 0.:
            pposH = 0.
        self.pposH = pposH
        self.pposL = pposL

    def GammaDist(self,gamma):
        return ((self.be**self.al)/gammaFunc(self.al))*(gamma**(self.al-1))*mpmath.exp(-self.be*gamma)

    def PiP0(self,gamma):
        U = 4*self.theta_f*self.Lf/(2.*self.NN)
        R = 2*self.Lf*self.rho/(2.*self.NN)
        return self.GammaDist(gamma)*mpmath.exp(-(self.GammaDist(gamma)*U/(2.*self.NN))/(gamma/(self.NN+0.)+R/(2.*self.NN)))

    def intPiP0(self):
        ret = lambda gam: self.PiP0(gam)
        return quad(ret,0.,1000)[0]

    def calcBGam(self,L,alpha,beta,theta):
        # not sure whats going on here, seems to overestimate or under sometimes
        u = 2*theta/(2.*self.NN)
        r = self.rho/(2.*self.NN)

        a = -1.*(2**(alpha+1))*np.exp(L*self.NN*r*beta)*L*self.N*((1./(L*self.N*r))**(1-alpha))*u*(beta**alpha)
        b =  float(gammainc(1-alpha,L*self.NN*r*beta))
        #fudge = 1-gamInt.cdf(0.2,a=self.al2,scale=1./self.be2)
        #fudge = 1.
        fudge = 0.25

        c = np.exp(a*b*fudge)
 
        return c
        #return np.frompyfunc(mpmath.exp(-1.*(2**(1+alpha))*mpmath.exp(L*self.NN*r*beta)*L*self.N*((1./(L*self.N*r))**(1-alpha))*u*(beta**alpha)*float(gammainc(1-alpha,L*self.NN*r*beta))),1,1)

    def calcB(self,L,theta):
        t = -1.*self.gam_neg/(self.NN+0.)
        u = 2.*theta/(2.*self.NN)
        r = self.rho/(2.*self.NN)

        #return u*t/((t+r*L)**2)
        #return u*t/(2*(t+(1.-np.exp(-2*r*L))/2.)**2)
        
        # Nordborg models
        #####return u/(2*r*t*(1+((1.-np.exp(-2*r*L))/2.)*(1-t)/t)**2)
        #return u/(t*(1+((1-np.exp(-2.*r*L))/2.)*(1-t)/t)**2)
        #return u/(t*(1+r*L*(1-t)/t)**2)
        #####return u/(t*(1+(np.exp(-2*r*L)/2)/t)**2)

    def Br(self,Lmax,theta):
        t = -1.*self.gam_neg/(self.NN+0.)
        u = theta/(2.*self.NN)
        r = self.rho/(2.*self.NN)
        #return np.exp(-2.*quad(lambda L: self.calcB(L,theta), 0., Lmax)[0])
        #print np.exp(-2.*t*u*(polygamma(1,(r+t)/r)-polygamma(1,1+Lmax+t/r))/r**2), np.exp(-u*2.*Lmax/(Lmax*r+2*t))
        return np.exp(-4*u*Lmax/(2*Lmax*r+t))

    def get_B_vals(self):
        ret = []
        for i in xrange(20,50):
            L = int(round(1.3**i))
            ret.append(self.Br(L),L)
        return ret
    
    def set_Lf(self):
        Lf  = fsolve(lambda L: self.Br(L,self.theta_f)-self.B,100)
        self.Lf = int(round(Lf[0]))

    def set_theta_f(self):
        theta_f  = fsolve(lambda theta: self.Br(self.Lf,theta)-self.B,0.00001)
        self.theta_f = theta_f[0]
    
    def set_theta_f_gam(self):
        theta_f  = fsolve(lambda theta: self.calcBGam(self.Lf,self.al2,self.be2,theta)-self.B,0.0000001)
        self.theta_f = theta_f[0]

    def make_sim(self,aFlank=False):

        t_tot = self.theta_mid_neutral*2*self.L_mid+self.theta_f*2*self.Lf
        t_tot = t_tot/(2*self.L_mid+2*self.Lf)
        rmid = (self.theta_mid_neutral*2*self.L_mid/(self.theta_f*self.Lf))

        var = 0.001
        mean_gam_H = self.gH
        lpH = mean_gam_H/var
        apH = mean_gam_H*lpH

        mean_gam_L = self.gL
        lpL = mean_gam_L/var
        apL = mean_gam_L*lpL

        com = command.SFSCommand(skip=self.skip)
        if self.scratch:
            com.outdir = "/scratch/users/uricchio/mktest/sims"
 
        #com.prefix = self.pref+'.gam'+str(-self.gam_neg)+'.pi'+str(self.B)+'.alL'+str(self.alLow)+".gL"+str(self.gL)
        com.prefix = self.pref+'.pi'+str(self.B) #+'.alL'+str(self.alLow)
        #if self.pref == "unc":
        #    com.prefix = 'unc.gam'+str(-self.gam_neg)+'.pi'+str(self.B)+'.alL'+str(self.alLow)+".gL"+str(self.gL)
        
        rf = "/home/uricchio/recombs/NO_rec.L"+str(self.Lf)+".2L_mid"+str(self.L_mid)+".rec"
        if 'SHERLOCK' in os.environ and int(os.environ['SHERLOCK'])==2:
            rf = "/home/users/uricchio/recombs/NO_rec.L"+str(self.Lf)+".2L_mid"+str(self.L_mid)+".rec"
        try:
            os.stat(rf)
        except:
            p0 = self.Lf
            p1 = self.Lf+self.L_mid
            p2 = self.Lf+2*self.L_mid
            p3 = 2*self.Lf+2*self.L_mid
            rfh = open(rf,'w')
            rfh.write("4\n")
            rfh.write(str(p0)+" "+str(0.5)+"\n")
            rfh.write(str(p1)+" "+str(0.5000000001)+"\n")
            rfh.write(str(p2)+" "+str(0.5000000002)+"\n")
            rfh.write(str(p3)+" "+str(1)+"\n")
            rfh.close()

        com.line = ['1',str(self.nsim),'-N',str(self.N),'-n',str(self.n),'-t',str(t_tot)]
        com.line.extend(['-L','4',str(self.Lf),str(self.L_mid),str(self.L_mid),str(self.Lf),'-l','p','0.0','R','-A'])
        if aFlank and not self.gF:
            mean_gam = -self.gam_neg
            lL = mean_gam/var
            aL = mean_gam*lL
            pposF =  0.75*(rmid/2.)*self.pposH*0.05*self.Lf/self.L_mid

            com.line.extend(['-W','L','0','2',str(pposF),str(apH),str(lpH),str(aL),str(lL)])
        elif aFlank and self.gF:
        
            pposF =  0.75*(rmid/2.)*self.pposH*0.05*self.Lf/self.L_mid
            com.line.extend(['-W','L','0','2',str(pposF),str(apH),str(lpH),str(self.al2),str(self.be2)])

        elif self.gF and not aFlank:
            com.line.extend(['-W','L','0','2','0.',str(apH),str(lpH),str(self.al2),str(self.be2)])

        else:    
            com.line.extend(['-W','L','0','1',str(-self.gam_neg),'0','1'])
        if not self.neut_mid:
            com.line.extend(['-W','L','1','2',str(self.pposL),str(apL),str(lpL),str(self.al),str(self.be)])
            com.line.extend(['-W','L','2','2',str(self.pposH),str(apH),str(lpH),str(self.al),str(self.be)])
        
        if aFlank and not self.gF:
            mean_gam = -self.gam_neg
            lL = mean_gam/var
            aL = mean_gam*lL
            pposF =  0.75*(rmid/2.)*self.pposH*0.05*self.Lf/self.L_mid

            com.line.extend(['-W','L','3','2',str(pposF),str(apH),str(lpH),str(aL),str(lL)])
        elif aFlank and self.gF:
        
            pposF =  0.75*(rmid/2.)*self.pposH*0.05*self.Lf/self.L_mid
            com.line.extend(['-W','L','3','2',str(pposF),str(apH),str(lpH),str(self.al2),str(self.be2)])

        elif self.gF and not aFlank:
            com.line.extend(['-W','L','3','2','0.',str(apH),str(lpH),str(self.al2),str(self.be2)])

        else:    
            com.line.extend(['-W','L','3','1',str(-self.gam_neg),'0','1'])
        com.line.extend(['-v','L','0','1','-v','L','1',str(rmid/2.),'-v','L','2',str(rmid/2.),'-v','L','3','1'])
        com.line.extend(['-r','F',str(rf),str(self.rho)])
        com.line.extend(['-a','N','C','C','N'])
        com.line.extend(['-TE',str(self.TE)])
   
        if self.demog == True:
            com.line.extend(['-Td','4.595', str(self.expan),'-Tg','4.985986','242.36'])

        com.sfs_code_loc = '/home/uricchio/pop_gen_software/sfs_code/bin/sfs_code'
        if 'SHERLOCK' in os.environ and int(os.environ['SHERLOCK'])==2:
            com.sfs_code_loc = '/home/users/uricchio/pop_gen_software/sfs_code/bin/sfs_code'
        com.execute()

        if self.ABC:
            
            if not self.scratch:
                zf = os.path.join(os.getcwd(),"sims/"+com.prefix+"/"+com.prefix+"."+str(self.task_id)+".txt")
            else:
                zf = os.path.join("/scratch/users/uricchio/mktest","sims/"+com.prefix+"/"+com.prefix+"."+str(self.task_id)+".txt")
            #if 'SHERLOCK' in os.environ and int(os.environ['SHERLOCK'])==2:
            #    zf = "/home/users/uricchio/projects/mktest/adapter/ABC/sims/"+com.prefix+"/"+com.prefix+"."+str(self.task_id)+".txt"
        else:
            if not self.scratch:
                zf = os.path.join(os.getcwd(),"sims/"+com.prefix+"/"+com.prefix+"."+str(self.task_id)+".txt")
            else:
                zf = os.path.join("/scratch/users/uricchio/mktest","sims/"+com.prefix+"/"+com.prefix+"."+str(self.task_id)+".txt")
                 
            #if 'SHERLOCK' in os.environ and int(os.environ['SHERLOCK'])==2:
            #    zf = "/home/users/uricchio/projects/mktest/adapter/sims/"+com.prefix+"/"+com.prefix+"."+str(self.task_id)+".txt"
                

        p = subprocess.Popen(["gzip","-f",zf])

        p.wait()

    def cumuSfs(self,sfsTemp):
        out = [np.sum(sfsTemp)]
        for i in range(0,len(sfsTemp)):
            app = out[i]-sfsTemp[i]
            if app > 0.:
                out.append(out[i]-sfsTemp[i])
            else:
                out.append(0.) 
        return out

    def alx(self,gammaL,gammaH,pposL,pposH):
        ret = []
        fN = self.B*self.fixNeut()
        fNeg = self.B*self.fixNegB(0.5*pposH+0.5*pposL)
        fPosL = self.fixPosSim(gammaL,0.5*pposL)
        fPosH = self.fixPosSim(gammaH,0.5*pposH)
        neut = self.cumuSfs(self.DiscSFSNeutDown())
        selH = self.cumuSfs(self.DiscSFSSelPosDown(gammaH,pposH))
        selL = self.cumuSfs(self.DiscSFSSelPosDown(gammaL,pposL))
        selN = self.cumuSfs(self.DiscSFSSelNegDown(pposH+pposL))        

        sel = []
        for i in range(0,len(selH)):
            sel.append((selH[i]+selL[i])+selN[i])
        for i in range(0,self.nn-1):
            ret.append(float(1. - (fN/(fPosL + fPosH+  fNeg+0.))* sel[i]/neut[i]))
        return ret
    
    def alx_noCumu(self,gammaL,gammaH,pposL,pposH):
        ret = []
        fN = self.B*self.fixNeut()
        fNeg = self.B*self.fixNegB(0.5*pposH+0.5*pposL)
        fPosL = self.fixPosSim(gammaL,0.5*pposL)
        fPosH = self.fixPosSim(gammaH,0.5*pposH)
        neut = self.DiscSFSNeutDown()
        selH = self.DiscSFSSelPosDown(gammaH,pposH)
        selL = self.DiscSFSSelPosDown(gammaL,pposL)
        selN = self.DiscSFSSelNegDown(pposH+pposL)        

        sel = []
        for i in range(0,len(selH)):
            sel.append((selH[i]+selL[i])+selN[i])
        for i in range(0,self.nn-1):
            ret.append(float(1. - (fN/(fPosL + fPosH+  fNeg+0.))* sel[i]/neut[i]))
        return ret

    def alx_nopos(self,gammaL,gammaH,pposL,pposH):
        # mean = (alpha/beta)
        # sdv  = alpha/beta^2
        # nal/nbe  = (alpha/beta)/2
        # nal/nbe^2 = sqrt(alpha)/beta^2/2
        ret = []
        fN = self.B*self.fixNeut()*(self.theta_mid_neutral/2.)*self.TE*self.NN
        fNeg = self.B*(self.theta_mid_neutral/2.)*self.TE*self.NN*self.fixNegB(0.5*pposH+0.5*pposL)
        fPosL = self.fixPosSim(gammaL,0.5*pposL)*(self.theta_mid_neutral/2.)*self.TE*self.NN
        fPosH = self.fixPosSim(gammaH,0.5*pposH)*(self.theta_mid_neutral/2.)*self.TE*self.NN
        
        neut = self.cumuSfs(self.DiscSFSNeutDown())
        selN = self.cumuSfs(self.DiscSFSSelNegDown(pposL+pposH))

        sel = []
        for i in range(0,len(selN)):
            sel.append(selN[i])
        for i in range(0,self.nn-1):
            ret.append(float(1. - (fN/(fPosL + fPosH+  fNeg+0.))* sel[i]/neut[i]))
        return ret
    
    def alx_nopos_noCumu(self,gammaL,gammaH,pposL,pposH):
        # mean = (alpha/beta)
        # sdv  = alpha/beta^2
        # nal/nbe  = (alpha/beta)/2
        # nal/nbe^2 = sqrt(alpha)/beta^2/2
        ret = []
        fN = self.B*self.fixNeut()*(self.theta_mid_neutral/2.)*self.TE*self.NN
        fNeg = self.B*(self.theta_mid_neutral/2.)*self.TE*self.NN*self.fixNegB(0.5*pposH+0.5*pposL)
        fPosL = self.fixPosSim(gammaL,0.5*pposL)*(self.theta_mid_neutral/2.)*self.TE*self.NN
        fPosH = self.fixPosSim(gammaH,0.5*pposH)*(self.theta_mid_neutral/2.)*self.TE*self.NN
        
        neut = self.DiscSFSNeutDown()
        selN = self.DiscSFSSelNegDown(pposL+pposH)

        sel = []
        for i in range(0,len(selN)):
            sel.append(selN[i])
        for i in range(0,self.nn-1):
            ret.append(float(1. - (fN/(fPosL + fPosH+  fNeg+0.))* sel[i]/neut[i]))
        return ret

    def print_alx(self,gammaL,gammaH,pposL,pposH):
        out_nopos = self.alx_nopos(gammaL,gammaH,pposL,pposH)
        out = self.alx(gammaL,gammaH,pposL,pposH)
        for i in range(0,len(out)):
            print i+1, out[i], 'c', 'a'
            print i+1, out_nopos[i], 'c','np'
 
    def print_alx_noCumu(self,gammaL,gammaH,pposL,pposH):
        out_nopos = self.alx_nopos_noCumu(gammaL,gammaH,pposL,pposH)
        out = self.alx_noCumu(gammaL,gammaH,pposL,pposH)
        for i in range(0,len(out)):
            print i+1, out[i], 'c', 'a'
            print i+1, out_nopos[i], 'c','np'
    
    def print_alpha_quantities(self,gammaL,gammaH,pposL,pposH):
        fN = self.B*self.fixNeut()*(self.theta_mid_neutral/2.)*self.TE*self.NN
        fNeg = self.B*(self.theta_mid_neutral/2.)*self.TE*self.NN*self.fixNegB(0.5*pposH+0.5*pposL)
        fPosL = self.fixPosSim(gammaL,0.5*pposL)*(self.theta_mid_neutral/2.)*self.TE*self.NN
        fPosH = self.fixPosSim(gammaH,0.5*pposH)*(self.theta_mid_neutral/2.)*self.TE*self.NN
        #neut = self.cumuSfs(self.DiscSFSNeutDown())
        #selH = self.cumuSfs(self.DiscSFSSelDown(gammaH,0.))
        #selL = self.cumuSfs(self.DiscSFSSelDown(gammaL,0.))

        print fN, fPosL, fNeg

    def print_alpha_SFS_quantities(self,gammaL,gammaH,pposL,pposH):
        neut = self.cumuSfs(self.DiscSFSNeutDown())
        selN = self.cumuSfs(self.DiscSFSSelNegDown(pposL))
        selP = self.cumuSfs(self.DiscSFSSelPosDown(gammaL,pposL))

        for thing in neut:
            print thing,
        print
        for thing in selN:
            print thing,
        print 
        for thing in selP:
            print thing,
        print 

