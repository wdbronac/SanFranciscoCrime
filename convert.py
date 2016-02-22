import csv, sqlite3

con = sqlite3.connect("train.db")
cur = con.cursor()
cur.execute("CREATE TABLE t (Dates, Category, Descript, DayOfWeek, PdDistrict, Resolution, Address, X, Y);")
with open('../data/train.csv','rb') as csvfile: 
    # csv.DictReader uses first line in file for column headings by default
    dr = csv.DictReader(csvfile) # comma is default delimiter
    to_db = [(i['Dates'],i['Category'],i['Descript'],i['DayOfWeek'],i['PdDistrict'],i['Resolution'],i['Address'],i['X'],i['Y']) for i in dr]

cur.executemany("INSERT INTO t (Dates, Category, Descript, DayOfWeek, PdDistrict, Resolution, Address, X, Y) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
con.commit()
print"baby it's done";
con.close()

