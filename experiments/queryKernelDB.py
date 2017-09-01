#! /usr/bin/env python

import sqlite3
import sys
import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "dnn_kernel.db")
conn = sqlite3.connect(db_path)

table_name = 'kernel_dict'

c = conn.cursor()

if len(sys.argv) == 1:
  print "batch size need to be specified!!!"
  exit()

batchsize = sys.argv[1]

c.execute("SELECT DISTINCT kernelname FROM "+table_name+" WHERE batchsize = "+batchsize)
kernel_name_list = c.fetchall()
for kernel in kernel_name_list:
  c.execute("select invocationorder from kernel_dict where kernelname = '"+kernel[0]+"' and batchsize = "+batchsize)
  print kernel[0]+","+str(max(c.fetchall())[0])

