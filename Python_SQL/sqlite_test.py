import sqlite3
import time
import re

class Operate_DB():
    def __init__(self,cur,table_name):
        self.cur = cur
        self.table_name = table_name
        self.cur.execute('pragma table_info(\'%s\')' % self.table_name)
        self.column_info = []
        self.primary_key = []
        self.column_name = []
        self.column_type = []

        table_info = self.cur.fetchall()
        for row in table_info:
            if row[5] == 1:
                self.primary_key.append(row[5])
            self.column_info.append(row[1:])
            self.column_name.append(row[1])
            self.column_type.append(row[2])

    def show_info(self):
        print(self.column_info)
        print(self.column_name)
        print(self.column_type)

    def edit_input_value_list(self,values):
        values_for_sql = ''
        for i, val in enumerate(values):
            if self.column_type[i+1] == 'INTEGER':
                values_for_sql += str(val) + ','
            elif self.column_type[i+1] == 'STRING':
                values_for_sql += ('\'' + str(val) + '\',')
            elif self.column_type[i+1] == 'BOOL':
                values_for_sql += str(val) + ','

        return values_for_sql[:-1]

    def get_curr_maxid(self):
        flag,curr_maxid_str = self.run_query('select max(id) from %s as tmp' % self.table_name)
        if flag:
            return int(re.sub("\\D", "", str(curr_maxid_str)))
        else:
            return -1


    def run_query(self,sql_sentence):
        try :
            self.cur.execute(sql_sentence)
            return True , self.cur.fetchall()
        except:
            return False , -1

    def show_table(self):
        sql = "select * from stock"
        result_flag,rows = self.run_query(sql)
        if result_flag:
            for row in rows:
                print(row)

    def add_row(self,values):
        val_sql = self.edit_input_value_list(values)
        curr_maxid = self.get_curr_maxid()
        sql = 'insert into stock (' + ','.join(map(str,self.column_name)) + ') values(' + str(curr_maxid) +  ' + 1,' + val_sql + ')'
        print(sql)
        result_flag,rows = self.run_query(sql)
        if not result_flag:
            print('ERROR')
        #print(result_flag)
        #show_all(cur)

    def edit(self,column='',value='',name=''):
        sql = 'update stock set ' +  str(column)  + ' = ' + str(value) + ' where name = \'' + str(name) + '\''
        result_flag,rows = self.run_query(sql)
        if not result_flag:
            print('ERROR')

    def add_column(self,column_name='',column_dtype='INTEGER'):
        sql = 'alter table %s add %s %s ' % (self.table_name,column_name,column_dtype)
        result_flag,rows = self.run_query(sql)
        if not result_flag:
            print('ERROR')

    def search_column(self,column_name):
        sql = 'select %s from %s' % (column_name,self.table_name)
        result_flag,rows = self.run_query(sql)
        if not result_flag:
            print('ERROR')


    def change_column(self,column_name,new_column_name,dtype):
        sql = 'ALTER TABLE %s CHANGE COLUMN %s %s %s' % (self.table_name,column_name,new_column_name,dtype)
        result_flag,rows = self.run_query(sql)
        if not result_flag:
            print('ERROR')
# TEST.dbを作成する
# すでに存在していれば、それにアスセスする。
dbname = 'TEST.db'
conn = sqlite3.connect(dbname)


# sqliteを操作するカーソルオブジェクトを作成
cur = conn.cursor()

# personsというtableを作成してみる
# 大文字部はSQL文。小文字でも問題ない。
#cur.execute('CREATE TABLE stock (id INTEGER PRIMARY KEY,name STRING, count INTEGER)')

#cur.execute('CREATE TABLE stock (id INTEGER PRIMARY KEY,name STRING, count INTEGER)')

#add()
#print('ok')
#add(cur,'cable',20)
#show_all(cur)

#cur.execute('pragma table_info(\'stock\')')
#rows = cur.fetchall()

#for row in rows:
#    print(row)


db = Operate_DB(cur,'stock')
db.show_info()
db.show_table()
#values = ['mouse',30]
#db.add_row(values)
#db.edit('count',10,'ssd')
#db.add_column('place','STRING')
db.show_table()

# データベースへコミット。これで変更が反映される。


while True:
    query_str = input() 
    if query_str == 'exit sqlite3':
        break
    else:
        db.run_query(self,query_str)

conn.commit()
conn.close()
