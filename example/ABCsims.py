from MKinfer import inferParams
import sys

test = inferParams.inferTools(nsims=1000,Bfile='../../ABC/mk_with_positions_BGS.txt',testsim = False,pref='ABC',ABCsumsdir='ABC_sums_neg_new',datadir='../../ABC/output2')
test.simulate()
