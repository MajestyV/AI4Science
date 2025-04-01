# import sqlite3  # 导入sqlite3模块 (使用sqlite3模块可以连接到一个DB文件)
from ase.db import connect

working_loc = 'Lingjiang'

saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/datasets'}

if __name__ == '__main__':
    dataset_title = 'iso17'

    with connect(f'{saving_dir_dict[working_loc]}/{dataset_title}/reference.db') as conn:
        # 读取表格
        for row in conn.select(limit=2):
            print(row.toatoms())
            print(row['positions'])
            print(row['total_energy'])
            print(row.data['atomic_forces'])

    # conn = sqlite3.connect(f'{saving_dir_dict[working_loc]}/{dataset_title}/reference.db')
    # c = conn.cursor()  # 创建一个游标对象, 用于执行SQL语句

    # cursor = conn.execute('SELECT * FROM table_name')

    # 看一看DB文件里有哪些表
    # c.execute("select * from sqlite_master").fetchall()

    # c.close()  # 关闭游标
    # conn.close()  # 关闭数据库连接