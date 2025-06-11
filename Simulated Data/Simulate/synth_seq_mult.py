#!/usr/bin/python

import biolib as bio
import random
import sys
import numpy

basestr = "ACGT"

def random_wm(lwm_l=10,lwm_h=10,dir0=0.5):
    # core lwm_l has dirichlet parameter dir, from where it increases linearly to 10 at the flank

    nflank_l = (lwm_h - lwm_l)//2
    nflank_r = lwm_h - lwm_l - nflank_l

    if nflank_l > 0:
        flank_grad = (20.0/dir0)**(1.0/nflank_l)
    elif nflank_r > 0:
        flank_grad = (20.0/dir0)**(1.0/nflank_r)
    else:
        flank_grad = 0.0000001
    wm = []

    dir = 20.0
    for n in range(nflank_l):
        wm += [[x for x in numpy.random.dirichlet([dir,dir,dir,dir])]]
        dir /= flank_grad

    dir = dir0
    for n in range(lwm_l):
        wm += [[x for x in numpy.random.dirichlet([dir,dir,dir,dir])]]
    for n in range(nflank_r):
        dir *= flank_grad
        wm += [[x for x in numpy.random.dirichlet([dir,dir,dir,dir])]]
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


def sample_seq(lseq,wmlist,embedlist,rclist,randvec):
    embedindex = [n for n in embedlist]
    random.shuffle(embedindex)
    header = ""
    lens = [len(wmlist[n]) for n in embedindex]
    lseq_left = lseq-sum(lens)
    breakpts = []
    while len(breakpts) < len(embedindex):
        k = random.randint(2,lseq_left-2)
        if k not in breakpts:
            breakpts += [k]
    breakpts.sort()
    wmpos = [b for b in breakpts]
    for n in range(1,len(wmpos)):
        wmpos[n] += sum(lens[0:n])
    breakpts += [lseq_left]
    for n in range(len(breakpts)-1,0,-1):
        breakpts[n] -= breakpts[n-1]
    s_bg = ["".join([sample_from_vec(randvec) for n in range(x)]) for x in breakpts]
    seq_out = s_bg[0]
    for n in range(len(embedindex)):
        wmseq = sample_from_wm(wmlist[embedindex[n]])
        if rclist[n]==1:
            wmseq = bio.revcomp(wmseq)
        seq_out += wmseq+s_bg[n+1]
        header += " | wm# "+str(embedindex[n])+" at "+str(wmpos[n])+" seq "+wmseq+ ["","(revcomp)"][rclist[n]]

    return header, seq_out


def printhelp():
    print("""
Usage: synth_seq_mult.py [options]

Generates sequences each with K distinct motifs embedded, from a set of N
weight matrices.  The motif subsets are taken randomly from M architectures.
For example, with 5 matrices and 3 architectures, architecture 1 could contain
motifs 1, 3, 4; arch 2 could have 2, 3, 4; arch 3 could have 1, 3, 5.

Outputs to stdout.

Options:
-N int         : number of distinct weight matrices (default 10)
-M int         : number of architectures (default 3)
-K int         : number of motifs per architecture (default 3)
-l int[,int]   : length of weight matrix (default 10); if second int specified, length
                 "fades out" from core d-value to d=10.0, over this range
-L int         : length of sequence (default 300)
-n int         : number of output sequences (default 1000)
-d float       : hyperparameter (pseudocount) of Dirichlet distribution from which WM is
                 sampled (default 0.2)
-r             : disable reverse complement instances (default: include reverse complements)
-b floatlist   : comma-separated list of background probabilities (default 0.25,0.25,0.25,0.25)
""")




Nwm = 10
March = 3
Kmotifs = 3
nseqs = 1000
rc = True
lseq = 300
lwm_l = 10
lwm_h = 10
dirpam = 0.2
randvec = [0.25,0.25,0.25,0.25]

import sys,getopt
optlist,args = getopt.getopt(sys.argv[1:],"N:M:K:l:L:n:d:rb:h")
for o,v in optlist:
    if o=="-N":
        Nwm = int(v)
    elif o=="-n":
        nseqs = int(v)
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
    elif o == "-K":
        Kmotifs = int(v)
    elif o == "-M":
        March = int(v)
    elif o == "-b":
        randvec = [float(x) for x in v.split(",")[0:4]]
        randvec = [x/sum(randvec) for x in randvec]
    else:
        printhelp()
        sys.exit()

outfilename = args[0]

wmlist = [random_wm(lwm_l,lwm_h,dirpam) for n in range(Nwm)]

archlist = [random.sample(range(Nwm),Kmotifs) for n in range(March)]


f = open(outfilename+".fa",'w')
for ns in range(nseqs):
    embedlist = random.sample(archlist,1)[0]
    if rc:
        rclist = [random.randint(0,1) for n in range(Kmotifs)]
    else:
        rclist = [0 for n in range(Kmotifs)]

    h,s = sample_seq(lseq,wmlist,embedlist,rclist,randvec)
    f.write(">"+str(ns+1)+h+"\n")
    f.write(s+"\n")

f.close()
f = open(outfilename+".tr","w")

for n in range(len(wmlist)):
    wm = wmlist[n]
    name = "wm"+str(n)
    bio.write_transfac(name,wm,f)
f.close()
