#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:23:22 2017

@author: cmendezg
"""

def tic():
    import time
    tic = time.time()
    return tic

def toc(tic):
    import time
    now = time.time()
    toc=now - tic
    return toc

def printelapsedtime(toc):
    m, s = divmod(toc, 60)
    h, m = divmod(m, 60)
    tocstr= ("%d:%02d:%02d" % (h, m, s))
    print (tocstr)
    return tocstr
    
    