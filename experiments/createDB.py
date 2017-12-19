#! /usr/bin/env python

import sqlite3
import csv

conn = sqlite3.connect('alexnet_kernel.db')
table_name = 'alexnet_kernel_dict'

c = conn.cursor()

# Create table
c.execute("CREATE TABLE "+table_name+\
          " (arch, batchsize, layername, propagation, kernelname, fullname, invocationorder)")

# Insert a row of data
reader = csv.reader(open('layer_kernel_dict.csv', 'rb'))
print "Table Name: ", table_name
for row in reader:
  content = "'"+row[0]+"',"+row[1]+",'"+row[2]+"','"+row[3]+"','"+row[4]+"','"+row[5]+"',"+row[6]
  print content
  c.execute("INSERT INTO "+table_name+" VALUES ("+content+")")

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
