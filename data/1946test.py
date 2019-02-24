import csv

with open('scotusjustice1946.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        print(row)

csvFile.close()



# pull out the filters
#
# make 116 separate arrys
#
#
# each array has the value repeating in the same order for each case they did in filters order
