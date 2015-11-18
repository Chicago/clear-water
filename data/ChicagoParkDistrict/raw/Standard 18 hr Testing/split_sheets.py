from __future__ import print_function
import xlrd
import csv
import os

def ascii_damnit(x):
	try:
		return x.encode(errors='ignore')
	except AttributeError :
		return x


for filename in os.listdir(os.getcwd()):
	if filename.endswith(".xls") or filename.endswith(".xlsx"):
		# Open the workbook
		xl_workbook = xlrd.open_workbook(filename)

		# List sheet names, and pull a sheet by name
		#
		sheet_names = xl_workbook.sheet_names()
		print(filename)


		for sheet in sheet_names:
			xl_sheet = xl_workbook.sheet_by_name(sheet)
			csv_name = filename.split('.')[0] + '_' + sheet + '.csv'
			with open(csv_name, 'wb') as out_file:

				writer = csv.writer(out_file)
				for row_idx in range(xl_sheet.nrows):
					writer.writerow( [sheet]+[ascii_damnit(col.value) for col in xl_sheet.row(row_idx)] )

