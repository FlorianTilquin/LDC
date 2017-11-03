#! /usr/bin/env python3
# -*- encoding:utf-8 -*-
#
# Ci-gît mon respect, tué par des requêtes SQL...

import csv
import sqlite3 as lite

conn = lite.connect('database.sqlite')
c = conn.cursor()

c.execute('SELECT `rowid`,* FROM `Match` WHERE `country_id` LIKE \'4769\' ORDER BY `rowid` ASC;')
# country_id <=> ligue_id, i.e. pas de match de coupe
descr = c.description
champs = [d[0] for d in descr]
# print(champs) # si tu veux voir sur quoi on peut tabuler

rows = c.fetchall()
csvWriter = csv.writer(open("table.csv", "w"))
csv.writerow(champs) #adapter si les champs on été réduits
csv.writerows(rows)

