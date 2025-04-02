import os
# import sqlite3  # 导入sqlite3模块 (使用sqlite3模块可以连接到一个DB文件)
from ase.db import connect
from ase.io import read, write
from ase.visualize import view

working_loc = 'Lingjiang'

dataset_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/datasets'}

saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir'}

# 参考文献：
# [1] https://wiki.fysik.dtu.dk/ase/ase/db/db.html#ase-db
# [2] https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write

if __name__ == '__main__':
    dataset_title = 'iso17'

    with connect(f'{dataset_dir_dict[working_loc]}/{dataset_title}/reference.db') as conn:

        # 读取表格
        for row in conn.select(limit=4000):
            config = row.toatoms()  # 将行转换为ASE原子对象
            config_id = row['id']  # 获取结构的id

            # print(row.toatoms())
            print(config)
            print(row['id'])
            print(row['positions'])
            print(row['total_energy'])
            print(row.data['atomic_forces'])

            # view(config, viewer='VMD')  # 使用ASE的可视化工具查看原子结构

            for fmt in ['png', 'cif', 'xyz']:
                os.makedirs(f'{saving_dir_dict[working_loc]}/{dataset_title}/{fmt}', exist_ok=True)  # 创建子目录用于存放数据
                write(f'{saving_dir_dict[working_loc]}/{dataset_title}/{fmt}/{config_id}.{fmt}', config, format=fmt)

    # conn = sqlite3.connect(f'{saving_dir_dict[working_loc]}/{dataset_title}/reference.db')
    # c = conn.cursor()  # 创建一个游标对象, 用于执行SQL语句

    # cursor = conn.execute('SELECT * FROM table_name')

    # 看一看DB文件里有哪些表
    # c.execute("select * from sqlite_master").fetchall()

    # c.close()  # 关闭游标
    # conn.close()  # 关闭数据库连接