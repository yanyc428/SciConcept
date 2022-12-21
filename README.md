# Sci-Concept

## 一、安装
Sci-Concept提供了一键安装方式，在命令行输入以下命令即可完成安装
```bash
pip install .
```

## 二、使用
在安装后，直接调用以下命令即可进入sci-concept终端：
```bash
sci-concept
```

进入终端后可以用`h`或`help`命令查看程序的帮助：

```
sci-concept >> help
领域术语识别与面向技术主题的术语规范化研究
Commands:     
    help     (h)  查看帮助
    quit     (q)  退出系统
    datasets (d)  导入数据集
    tree     (t)  术语规范化
```

使用`datasets`或`d`命令查看导入数据集功能的相关信息
``` 
sci-concept >> datasets
Usage: sci-concept-datasets [OPTIONS] COMMAND [ARGS]...

  导入数据集

Options:
  -h, --help  Show this message and exit.

Commands:
  add   添加数据集
  list  显示已导入的数据集
```

使用`datasets add`命令添加数据集
``` 
sci-concept >> datasets add -h
Usage: sci-concept-datasets add [OPTIONS]

  添加数据集

Options:
  -n, --name TEXT      数据集名称  [required]
  -d, --data TEXT      文档文件的路径，csv格式，至少包含ABSTRACT字段，若不实用自建术语表，还需要提供AUTHOR_KEYW
                       ORDS字段  [required]
  -k, --keywords TEXT  自建术语表的路径，txt格式，每行一个词语
  -h, --help           Show this message and exit.
```
下面给出了导入数据集的命令
```
sci-concept >> datasets add -n name_of_dataset -d path_to_data_csv -k path_to_term_txt
```
使用`datasets list`命令查看当前已导入的数据集
``` 
sci-concept >> datasets list
     Name                 Date       Papers     Keywords   Valid_Keywords Status    
   1 dataset_24K_AUTO     2022-12-15 241976     10000      9832           导入完成      
   2 dataset_3.6K_AUTO    2022-12-15 35695      10000      9621           导入完成      
   3 dataset_3.6K_6K      2022-12-15 35695      6641       6374           导入完成      
   4 dataset_8K_8K        2022-12-20 6511       8560       6938           导入完成      
   5 dataset_6.5K_7K      2022-12-20 6511       8560       5500           导入完成      
   6 dataset_6.5K_AUTO    2022-12-20 6511       9427       6416           导入完成      
   7 test                 2022-12-21 6511       8560       5500           导入完成    
```

使用`tree`或`t`命令查看术语规范化功能的相关信息
```
sci-concept >> tree
Usage: sci-concept-tree [OPTIONS] COMMAND [ARGS]...

  术语规范化

Options:
  -h, --help  Show this message and exit.

Commands:
  generate  生成层次分类体系
  search    术语规范化
```
使用`tree generate`生成层次概念体系
``` 
sci-concept >> tree generate -h
Usage: sci-concept-tree generate [OPTIONS]

  生成层次分类体系

Options:
  -n, --name TEXT  数据集名称，可使用datasets命令查看所有数据集
  -k INTEGER       层级最大节点数
  -h, --help       Show this message and exit.
```
下面给出了一个示例
``` 
tree generate -n name_of_dataset -k 15
```
使用`tree search`在生成的层次概念体系中检索
``` 
sci-concept >> tree search -h
Usage: sci-concept-tree search [OPTIONS] WORD

  术语规范化

Options:
  -s, --semantic INTEGER  使用语义检索
  -h, --help              Show this message and exit.
```
下面给出了一个检索的例子
``` 
tree search term_to_search -s 3
```