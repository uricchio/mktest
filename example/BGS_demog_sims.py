#!/usr/bin/python
#$ -e sim.div.log
#$ -o sim.div.log
#$ -S /usr/bin/python
#$ -cwd
#$ -r yes
#$ -l h_rt=168:00:00
#$ -t 101-200

from adapter import mkcalc
from scipy.optimize import fsolve
import sys

BSim  = float(sys.argv[1])
gam_neg = -1.*float(sys.argv[2])
gL = float(sys.argv[3])

# stuff
adap = mkcalc.AsympMK(B=BSim,gam_neg=gam_neg,theta_f=0.001,alLow=0.2,alTot=0.4,gH=500,neut_mid=False,L_mid=501,Lf=2*10**5,N=500,n=661,nsim=100,pref="DE2",gL=gL,demog=True,ABC=True)

# get correct theta, ppos values
adap.set_theta_f()
theta_f = adap.theta_f
adap.B = 0.999
adap.set_theta_f()
adap.setPpos()
adap.theta_f = theta_f
adap.B= BSim

# run the simulation
adap.make_sim()
