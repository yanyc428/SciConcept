import datetime
import subprocess

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from database import *
from tree import ConceptTree
import os
from pydantic import BaseModel
from typing import Union

app = FastAPI()  # 实例化fastapi
app.mount('/static', StaticFiles(directory="static"), 'static')

# 使用中间件以拓展跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trees = dict()


@app.get('/api/datasets')
async def get_available_datasets():
    session = Session()
    datasets = session.query(Dataset).all()
    session.close()
    return datasets


@app.get('/api/datasets/done')
async def get_available_datasets():
    session = Session()
    datasets = session.query(Dataset).filter_by(status='导入完成').all()
    session.close()
    return [dataset.name for dataset in datasets]


@app.post('/api/tree')
async def export_tree(name: str, k: int):
    if (name, k) in trees:
        return trees[(name, k)].export()
    tree = ConceptTree(name, k)
    tree.construct()
    trees[(name, k)] = tree
    return tree.export()


@app.post('/api/query')
async def query_tree(name: str, k: int, word: str):
    tree = trees[(name, k)]
    try:
        for keyword, _ in tree.dataset.w2v.most_similar(word, topn=10):
            if keyword in tree.dataset.keywords:
                return keyword
    except KeyError:
        return ''
    return ''


@app.get('/', response_class=HTMLResponse)
async def index():
    html_file = open("index.html", 'r').read()
    return html_file


class Form(BaseModel):
    name: str
    use_term: bool
    data: UploadFile
    term: UploadFile = None


@app.post('/api/datasets/add', response_class=JSONResponse)
async def upload_data(name: str, data: UploadFile, term: Union[UploadFile, None] = None):
    if not os.path.exists(os.path.join('checkpoints', name)):
        os.makedirs(os.path.join('checkpoints', name))

    with open(os.path.join('checkpoints', name, 'data.csv'), 'wb') as f:
        f.write(data.file.read())

    if term is not None:
        with open(os.path.join('checkpoints', name, 'keywords.txt'), 'wb') as f:
            f.write(term.file.read())

    dataset = Dataset(name=name, papers=0, keywords=0, real_keywords=0, create_date=datetime.datetime.now(),
                      status='正在创建')
    session = Session()
    session.add(dataset)
    session.commit()
    session.close()

    subprocess.Popen(['python', 'data.py', '-ckpt', name])
    return JSONResponse('success', status_code=200)


@app.get('/api/datasets/delete')
async def delete_data(_id: int):
    session = Session()
    d = session.query(Dataset).filter_by(id=_id).first()
    session.delete(d)
    session.commit()
    session.close()
    return 'success'
