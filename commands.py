# -*- coding: utf-8 -*-
"""
@Project    : SciConcept
@File       : commands
@Email      : yanyuchen@zju.edu.cn
@Author     : Yan Yuchen
@Time       : 2022/12/20 20:53
"""
import os


def main():
    os.system('python database.py')
    while True:
        option = input('sci-concept >> ')
        if option.strip() == '':
            continue
        options = option.strip().split()

        if options[0].startswith('help') or options[0].startswith('h'):
            print("""领域术语识别与面向技术主题的术语规范化研究
Commands:     
    help     (h)  查看帮助
    quit     (q)  退出系统
    datasets (d)  导入数据集
    tree     (t)  术语规范化""")
        elif options[0].startswith('quit') or options[0].startswith('q'):
            print('感谢使用!')
            exit(0)
        elif options[0].startswith('datasets') or options[0].startswith('d'):
            os.system('sci-concept-datasets ' + ' '.join(options[1:]))
        elif options[0].startswith('tree') or options[0].startswith('t'):
            os.system('sci-concept-tree ' + ' '.join(options[1:]))
        else:
            print("错误的参数!")
            print("""
Commands:     
    help     (h)  查看帮助
    quit     (q)  退出系统
    datasets (d)  导入数据集
    tree     (t)  术语规范化""")


if __name__ == '__main__':
    main()
