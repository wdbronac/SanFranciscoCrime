import csv, sqlite3

con = sqlite3.connect("train.db")
#cur = con.cursor()
#cur.execute("CREATE TABLE t (col1, col2,col3,col4,col5);")
#ts=con.execute('''SELECT * FROM t;''');
#for ts in t:
#	print ts

cursor = con.execute("SELECT Dates, Category, Descript, DayOfWeek, PdDistrict, Resolution, Address, X, Y  from t")
for row in cursor:
   print "Dates = ", row[0]
   print "Category = ", row[1]
   print "Descript = ", row[2]
   print "DayOfWeek = ", row[3]
   print "PdDistrict = ", row[4]
   print "Resolution = ", row[5]
   print "Address = ", row[6]
   print "X=", row[7]
   print "Y=", row[8], "\n"

print "La table est affichee baby !";
con.close()
