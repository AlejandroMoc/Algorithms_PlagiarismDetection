import random as rnd

def smp_lst():
    nmbrs_lst = [rnd.randint(1, 100) for _ in range(10)]
    srt_lst(nmbrs_lst)

def srt_lst(lst):
    lst.sort()
    prnt_lst(lst)

def prnt_lst(lst):
    for nmbr in lst:
        print(nmbr)

smp_lst()