#!/usr/bin/env python3

import bio_lib as bio
import random
import sys
import numpy

basestr = "ACGT"

def random_wm(lwm_l=10,lwm_h=10,dir0=0.5):
    # core lwm_l has dirichlet parameter dir, from where it increases linearly to 10 at the flank

    # nflank_l = (lwm_h - lwm_l)//2
    # nflank_r = lwm_h - lwm_l - nflank_l

    # if nflank_l==0:
    #     nflank_l=1
    # if nflank_r==0:
    #     nflank_r=1
    # flank_grad = (20.0/dir0)**(1.0/nflank_l)
    wm = []

    # dir = 20.0
    # for n in range(nflank_l):
    #     wm += [[x for x in numpy.random.dirichlet([dir,dir,dir,dir])]]
    #     dir /= flank_grad

    dir = dir0
    for n in range(lwm_l):
        wm += [[x for x in numpy.random.dirichlet([dir,dir,dir,dir])]]
    # for n in range(nflank_r):
    #     dir *= flank_grad
    #     wm += [[x for x in numpy.random.dirichlet([dir,dir,dir,dir])]]
    return wm


def sample_from_vec(v):
    r = random.random()
    n = 0
    t = v[0]
    while r>t and n<3:
        n += 1
        t += v[n]
    return basestr[n]

def sample_from_wm(wm):
    return "".join([sample_from_vec(v) for v in wm])


def choosewm(wmlist,wwm):
    rt = 0.0
    rrand = random.random()
    n = -1
    while rt <= rrand and n < len(wwm)-1:
        n += 1
        rt += wwm[n]
    return wmlist[n],n


def printhelp():
    print("""
Usage: synth_seq.py [options] basefilename

Generates sequences each with a motif embedded, from a set of n weight matrices
Outputs two files: basefilename.fa (sequence) and basefilename.txt (architectures)

Options:
-l int[,int]   : length of weight matrix (default 10); if second int specified, length
                 "fades out" from core d-value to d=10.0, over this range
-L int         : length of sequence (default 100)
-n int[,int]   : number of distinct weight matrices (default 2); if second int
                 specified, random number in this range
-N int         : number of output sequences (default 5000)
-d float       : hyperparameter (pseudocount) of Dirichlet distribution from which WM is
                 sampled (default 0.2)
-r             : disable reverse complement instances (default: include reverse complements)
-s int         : motif instance may be placed +- s basepairs from centre (default: 10)
-w float       : comma-separated list of probabilities for WMs (eg, 0.5,0.3 means first
                 WM will be used 50% of the time, second WM 30% of the time; remainder will be
                 used equally often)
""")

randvec = [1.0, 0.0, 0.0, 0.0]

nwm_l = 2
nwm_h = 2
wwm = []
nseqs = 5000
rc = True
shiftrange = 10
lseq = 100
lwm_l = 10
lwm_h = 10
dirpam = 0.2

import getopt
optlist,args = getopt.getopt(sys.argv[1:],"n:s:N:l:L:rd:w:h")
for o,v in optlist:
    if o=="-n":
        if "," in v:
            nwm_l, nwm_h = [int(x) for x in v.split(",")]
        else:
            nwm_l = nwm_h = int(v)
    elif o=="-N":
        nseqs = int(v)
    elif o=="-s":
        shiftrange = int(v)
    elif o=="-l":
        if "," in v:
            lwm_l,lwm_h = [int(x) for x in v.split(",")]
        else:
            lwm_l = lwm_h = int(v)
    elif o=="-L":
        lseq = int(v)
    elif o=="-r":
        rc = False
    elif o=="-d":
        dirpam = float(v)
    elif o == "-w":
        wwm = [float(x) for x in v.split(",")]
    else:
        printhelp()
        sys.exit()
if len(args)==0:
    printhelp()
    sys.exit()


outfilename = args[0]

nwm = random.randint(nwm_l,nwm_h)

if len(wwm)<nwm:
    restw = (1.0-sum(wwm))/(nwm-len(wwm))
    while len(wwm)<nwm:
        wwm += [restw]
wwm = [x/sum(wwm) for x in wwm]
sys.stderr.write(str(nwm)+" wms with weights "+str(wwm)+"\n")

wmlist = [random_wm(lwm_l,lwm_h,dirpam) for n in range(nwm)]

outf = open(outfilename+".fa","w")
outaf = open(outfilename+".txt","w")
countwm = [0 for n in range(nwm)]

for n in range(nseqs):
    shift = random.randint(-shiftrange,shiftrange)
    s = ""
    wm,nw = choosewm(wmlist,wwm)
    lwm = lwm_h
    padleft = (lseq-lwm)//2
    padright = lseq-padleft-lwm
    for m in range(padleft+shift):
        s += sample_from_vec(randvec)
    countwm[nw] += 1
    s += sample_from_wm(wm)
    for m in range(padright-shift):
        s += sample_from_vec(randvec)
    outf.write(">"+str(n+1)+" wm# " + str(nw+1) +" len "+str(lwm_l)+","+str(lwm_h))
    if rc==False or random.randint(0,1)==0:
        outf.write(" pos "+str(padleft+shift)+" (+)\n")
        outf.write(s+"\n")
        outaf.write(str(nw+1)+"\t"+str(n+1)+"\t"+s+"\t1.0\n")
    else:
        outf.write(" pos "+str(padright-shift)+" (-)\n")
        outf.write(bio.revcomp(s)+"\n")
        outaf.write(str(nw+1)+"\t"+str(n+1)+"\t"+bio.revcomp(s)+"\t1.0\n")
outf.close()
outaf.close()

outtf = open(outfilename+".tr","w")
for n in range(nwm):
    bio.write_transfac("WM_"+str(n+1)+"_("+str(countwm[n])+"_SEQS)",wmlist[n],outtf)

outtf.close()
