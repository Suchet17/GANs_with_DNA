from math import log
import gzip

codon_of_aa_standard = {'A':["GCA","GCC","GCG","GCT"],\
               'C':["TGC","TGT"],\
               'D':["GAC","GAT"],\
               'E':["GAA","GAG"],\
               'F':["TTC","TTT"],\
               'G':["GGA","GGC","GGG","GGT"],\
               'H':["CAC","CAT"],\
               'I':["ATA","ATC","ATT"],\
               'K':["AAA","AAG"],\
               'L':["TTA","TTG","CTA","CTC","CTG","CTT"],\
               'M':["ATG"],\
               'N':["AAC","AAT"],\
               'P':["CCA","CCC","CCG","CCT"],\
               'Q':["CAA","CAG"],\
               'R':["AGA","AGG","CGA","CGC","CGG","CGT"],\
               'S':["AGC","AGT","TCA","TCC","TCG","TCT"],\
               'T':["ACA","ACC","ACG","ACT"],\
               'V':["GTA","GTC","GTG","GTT"],\
               'W':["TGG"],\
               'Y':["TAC","TAT"],\
               '*':["TAA","TAG","TGA"] } # stop codons


codon_of_aa_ctg_clade = {}
for k in codon_of_aa_standard.keys():
    codon_of_aa_ctg_clade[k]=codon_of_aa_standard[k]

codon_of_aa_ctg_clade['L']=["TTA","TTG","CTA","CTC","CTT"]
codon_of_aa_ctg_clade['S']=["AGC","AGT","TCA","TCC","TCG","TCT","CTG"]

aa_of_codon_standard={}
for k in codon_of_aa_standard.keys():
    for c in codon_of_aa_standard[k]:
        aa_of_codon_standard[c] = k

aa_of_codon_ctg_clade={}
for k in codon_of_aa_ctg_clade.keys():
    for c in codon_of_aa_ctg_clade[k]:
        aa_of_codon_ctg_clade[c] = k



import re

def wvconsensus(w):
    c='n'
    wtot=w[0]+w[1]+w[2]+w[3]
    if w[0] > 0.51*wtot:
        c='A'
    elif w[1] > 0.51*wtot:
        c='C'
    elif w[2] > 0.51*wtot:
        c='G'
    elif w[3] > 0.51*wtot:
        c='T'
    elif ((w[0]+w[1]) > 3.0*wtot/4.0):
        c='M'
    elif ((w[2]+w[3]) > 3.0*wtot/4.0):
        c='K'
    elif ((w[0]+w[2]) > 3.0*wtot/4.0):
        c='R'
    elif ((w[1]+w[3]) > 3.0*wtot/4.0):
        c='Y'
    elif ((w[0]+w[3]) > 3.0*wtot/4.0):
        c='W'
    elif ((w[1]+w[2]) > 3.0*wtot/4.0):
        c='S'
    elif ((w[0]+w[1]+w[2]) > 7.0*wtot/8.0):
        c='V'
    elif ((w[0]+w[1]+w[3]) > 7.0*wtot/8.0):
        c='H'
    elif ((w[0]+w[2]+w[3]) > 7.0*wtot/8.0):
        c='D'
    elif ((w[3]+w[1]+w[2]) > 7.0*wtot/8.0):
        c='B'
    return c

def wmconsensus(wm):
    return "".join([wvconsensus(v) for v in wm])

def re_threecodons(aastr,codontable="normal"):
    if codontable=="ctg":
        codon_of_aa = codon_of_aa_ctg_clade
    else:
        codon_of_aa = codon_of_aa_standard
    restr = ""
    for c1 in codon_of_aa[aastr[0]]:
        for c2 in codon_of_aa[aastr[1]]:
            for c3 in codon_of_aa[aastr[2]]:
                restr += c1+c2+c3+"|"
    return restr[0:-1]

def re_twocodons(aastr,codontable="normal"):
    if codontable=="ctg":
        codon_of_aa = codon_of_aa_ctg_clade
    else:
        codon_of_aa = codon_of_aa_standard
    restr = ""
    for c1 in codon_of_aa[aastr[0]]:
        for c2 in codon_of_aa[aastr[1]]:
            restr += c1+c2+"|"
    return restr[0:-1]


def re_onecodon(aastr,codontable="normal"):
    if codontable=="ctg":
        codon_of_aa = codon_of_aa_ctg_clade
    else:
        codon_of_aa = codon_of_aa_standard
    restr = ""
    for c1 in codon_of_aa[aastr[0]]:
        restr += c1+"|"
    return restr[0:-1]



def extract_coding_seq(aas,nts,codontable="normal"):
    if codontable=="ctg":
        codon_of_aa = codon_of_aa_ctg_clade
        aa_of_codon = aa_of_codon_ctg_clade
    else:
        codon_of_aa = codon_of_aa_standard
        aa_of_codon = aa_of_codon_standard
    if len(nts) < 3*len(aas):
        return ""
    aas = aas.upper()
    nts = nts.upper()
    n_aa = 0
    n_nt = 0
    out_nts = ""
    while n_aa < len(aas) and n_nt < len(nts)-2  and \
              aa_of_codon[nts[n_nt:n_nt+3]]==aas[n_aa]:
        out_nts += nts[n_nt:n_nt+3]
        n_nt += 3
        n_aa += 1
    if n_aa == len(aas):
        return out_nts
    aas1 = aas[n_aa:]
    nts1 = nts[n_nt+1:]
    if len(aas1) < 3:
        return ""

    codonpat = re.compile(re_threecodons(aas1))
    nextmatch = codonpat.search(nts1)
    while nextmatch:
        n_nt = nextmatch.start()
        nts1 = nts1[n_nt:]

        out_nts_rest = extract_coding_seq(aas1,nts1)
        if out_nts_rest:
            return out_nts + out_nts_rest
        else:
            nts1 = nts1[1:]
            nextmatch = codonpat.search(nts1)
    return ""




def get_coords(h): # return embedded coords in form X:s..e or X:s-e if exist
    if h.startswith(">"):
        h = h[1:]
    words = h.split()
    for w in words:
        if ":" in w and (".." in w.split(":")[1] or "-" in w.split(":")[1]):
            if ".." in w.split(":")[1]:
                splitter = ".."
            else:
                splitter = "-"
            #get rid of trailing punctuation
            while not w[-1].isdigit():
                w = w[0:-1]
            ch = w.split(":")[0]
            s = w.split(":")[1].split(splitter)[0]
            e = w.split(":")[1].split(splitter)[1]
            if s.isdigit() and e.isdigit():
                return ch, int(s), int(e)
            else:
                continue
    return ()


def consensus(c1,c2,c3=""):
    if c1.isupper() or c2.isupper() or c3.isupper():
        upper = 1
    else:
        upper = 0
    c1 = c1.upper()
    c2 = c2.upper()
    if c1 == "U":
        c1 = "T"
    if c2 == "U":
        c2 = "T"
    if c1==c2 or c1 not in "ACGT" or c2 not in "ACGT":
        raise ValueError("At least two different nucleotides required")
    if c1 > c2:
        (c1,c2) = (c2,c1)
    if c3:
        c3 = c3.upper()
        if c3=="U":
            c3 = "T"
        if c1==c3 or c2==c3 or c3 not in "ACGT":
            raise ValueError("At least two different nucleotides required")
        s = c1+c2+c3
        if 'A' not in s:
            output = "B"
        elif 'C' not in s:
            output = "D"
        elif 'G' not in s:
            output = "H"
        elif ('T' not in s) and ('U' not in s):
            output = "V"

    elif c1=="A":
        if c2 =="G":
            output = "R"
        elif c2 == "C":
            output = "M"
        elif c2 in "UT":
            output = "W"
    elif c1=="C":
        if c2=="G":
            output = "S"
        elif c2 in "UT":
            output = "Y"
    elif c1=="G":
        if c2 in "UT":
            output = "K"
    if upper:
        return output
    else:
        return output.lower()

def read_one_wm(tflines,pseudocount):
    try:
        name = [l for l in tflines if l.startswith("NA")][0].split()[1]
    except IndexError:
        name = " "
    nums = [[(float(x)+pseudocount) for x in l.split()[1:5]] for l in tflines\
            if len(l)>2 and l[0].isdigit() and l[1].isdigit()]

    return (name,[[x/sum(l) for x in l] for l in nums])

def readtransfac(filename,pseudocount=0):
    f = open(filename,'r')
    flines = [l for l in f.read().split("//") if "NA" in l]
    f.close()
    return [read_one_wm(l.splitlines(),pseudocount) for l in flines]

def wm_transpose_norm(wm,pseudocount=0.0):
        new_wm = [[0.,0.,0.,0.] for x in wm[0]]
        for n in range(len(wm[0])):
            for m in range(4):
                try:
                    new_wm[n][m] = wm[m][n]
                except:
                    print(wm)
                    raise IndexError
        new_wm = [[(x+pseudocount)/(sum(l)+4.*pseudocount) for x in l] for l in new_wm]
        return new_wm


def readjaspar(filename,pseudocount=0.):
    wmlist = []
    currwm = []
    currname = ""
    f = open(filename,'r')
    for l in f.read().splitlines():
        if l.startswith(">"):
            if currname:
                wmlist += [(currname,wm_transpose_norm(currwm,pseudocount))]
                currname = ""
                currwm = []
            currname = l.split()[-1]
        elif len(l)<5:
            continue
        else:
            l = l.split("[")[1].split("]")[0]
            currwm += [[float(x) for x in l.split()]]
    wmlist += [(currname,wm_transpose_norm(currwm,pseudocount))]
    return wmlist


def readuniprobe(filename,pseudocount=0.):
    f = open(filename)
    wmlist = []
    currwm = []
    currname = ""
    for l in f.read().splitlines():
        if "Name" in l:
            if currname:
                wmlist += [(currname,wm_transpose_norm(currwm,pseudocount))]
                currname = ""
                currwm = []
            currname = l.split()[2]
        elif len(l)<5 or l.startswith("#"):
            continue
        else:
            currwm += [[float(x) for x in l.split()]]
    wmlist += [(currname,wm_transpose_norm(currwm,pseudocount))]
    return wmlist

def twodigitstr(n):
    if n<=9:
        return "0"+str(n)
    else:
        return str(n)

def wm_from_seqlist(seqlist, pscount=0.0):
    basestr = "ACGT"
    wm = [[len([s for s in seqlist if s[m]==basestr[n]]) for n in range(4)] for m in range(len(seqlist[0]))]
    wm = [[float(x+pscount)/(sum(w)+4*pscount) for x in w] for w in wm]
    return wm

def align_seqs_to_wm(seqlist,wm):
    outs = []
    for s in seqlist:
        sr = revcomp(s)
        score1 = wmscore(s,wm)
        score2 = wmscore(sr,wm)
        if score1 > score2:
            outs += [s]
        else:
            outs += [sr]
    return outs

def write_transfac(name,wm,filehandle):
    filehandle.write("NA\t"+name+"\n")
    filehandle.write("PO\tA\tC\tG\tT\n")
    for i in range(len(wm)):
        filehandle.write("\t".join([twodigitstr(i+1)] + [f"{x:.3f}" for x in wm[i]])+"\n")
    filehandle.write("//\n")


def write_full_transfac(wmlist,filename):
    f = open(filename,"w")
    for name,wm in wmlist:
        write_transfac(name,wm,f)
    f.close()

def write_mememotif_header(f,bgfreqs=[]):
    f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
    if bgfreqs:
        f.write("Background letter frequencies\nA "+str(bgfreqs[0])+" C "+str(bgfreqs[1])\
                +" G "+str(bgfreqs[2])+" T "+str(bgfreqs[3])+"\n\n")

def write_mememotif(name,wm,f,nsites=-1):
    f.write("MOTIF "+name+"\n")
    f.write("letter-probability matrix:")
    if nsites >= 0:
        f.write(" nsites= "+str(nsites))
    f.write("\n")
    for l in wm:
        l = [(x/sum(l)) for x in l]
        f.write(" ".join([str(x) for x in l])+"\n")
    f.write("\n")


def read_meme(filename,pscount=0.5):
    wmlist = []
    wm = []
    name = ""
    reading = 0
    for l in open(filename):
        if len(l) < 2:
            continue
        elif l.startswith("MOTIF"):
            name = l.split()[1]
            reading = 1
        elif l.startswith("letter"):
            nsites = int(l.split("nsites= ")[1].split()[0])
            reading = 2
        elif reading==2 and (l.split()[0].startswith("0") or l.split()[0].startswith("1")):
            wm += [[float(x) for x in l.split()]]
        else:
            if wm > []:
                wm = [[x/sum(l) for x in l] for l in wm]
                wm = [[(x*nsites + pscount) for x in l] for l in wm]
                wm = [[x/sum(l) for x in l] for l in wm]
                wmlist += [(name,wm)]
                name = ""
                wm = []
                reading = 0
    if wm > []:
        wm = [[x/sum(l) for x in l] for l in wm]
        wm = [[(x*nsites + pscount) for x in l] for l in wm]
        wm = [[x/sum(l) for x in l] for l in wm]
        wmlist += [(name,wm)]
    return wmlist


def readfasta(filename):
    fastalist=[]
    if filename.endswith(".gz"):
        f = gzip.open(filename,'rt')
    else:
        f = open(filename,'rt')
    fl = f.read().splitlines()
    f.close()

    currhead=""
    currseq=""
    for l in fl:
        if l=="":
            continue
        if l[0]==">":
            if currhead:
                fastalist += [(currhead,currseq)]
            currhead = l
            currseq=""
            continue
        if (l[0].upper()>='A' and l[0].upper()<='Z') or l[0]=='-':
            currseq += "".join(l.split())
    return fastalist+[(currhead,currseq)]


revcomp_char = {'A':'T','T':'A','C':'G','G':'C','R':'Y','Y':'R','M':'K',\
                'K':'M','S':'S','W':'W','B':'V','V':'B','D':'H','H':'D',\
                'N':'N'}

for c in "ATGCRYMKSWBDHVN":
    revcomp_char[c.lower()] = revcomp_char[c].lower()
revcomp_char["-"]="-"

def revcomp(s):
    s1 = [revcomp_char[c] for c in s]
    s1.reverse()
    return "".join(s1)

basehash={'A':0,'C':1,'G':2,'T':3,'S':4,'W':5,'R':6,'Y':7,'M':8,'K':9, \
          'B':10,'D':11,'H':12,'V':13,'N':14}

def scanwm(wm,seq):
    seqn = [basehash[c] for c in seq]
    best = -1000.0
    bestn = -1
    bestr = False
    lwm = len(wm)
    for n in range(len(seqn)-lwm+1):
        ll = 0.0
        llr = 0.0
        for m in range(lwm):
            ll += log(wm[m][seqn[n+m]]/0.25)
            llr += log(wm[m][3-seqn[n+lwm-m-1]]/0.25)
        if ll>best:
            best = ll
            bestn = n
            bestr = False
        if llr > best:
            best = llr
            bestn = n
            bestr = True
    return best, bestn, bestr


def writefasta_padded(fl, filename,L=1000):
    f = open(filename,"w")
    for h,s in fl:
       f.write(h+"\n"+s)
       for n in range(max(L-len(s),0)):
           f.write("N")
       f.write("\n")
    f.close()

def writefasta(fl, filename):
    f = open(filename,"w")
    for h,s in fl:
       f.write(h+"\n"+s+"\n")
    f.close()
