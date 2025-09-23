import asyncio
from fastapi import Request, FastAPI, BackgroundTasks, HTTPException
from schemas import TaskResponse, OptimizationRequest, StatusResponse
import datetime
import uuid
import logging
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List
app = FastAPI()
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

tasks_db = {}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def f(x):
    return str(x) + '12321321312'

@app.get('/')
async def base():
    return tasks_db

async def run_analyze(task_id, **kwargs):
    try:
        await asyncio.sleep(30)
        tasks_db[task_id]['status'] = 'DONE'
        tasks_db[task_id]['completed_at'] = datetime.datetime.now()
        tasks_db[task_id]['results'] = 'succes'

        logger.info(f"task {task_id} war completed correctly")

    except:
        logger.error(f"tast {task_id} was completed incorrectly")
        tasks_db[task_id]['status'] = 'FAILED'


@app.post("/new", response_model=TaskResponse)
async def create_optimization_task(
        req: OptimizationRequest,
        background_tasks: BackgroundTasks
):
    try:
        if not req.ddl:
            raise HTTPException(status_code=400, detail="DDL not found")
        if not req.queries:
            raise HTTPException(status_code=400, detail="Queries not found")
        # if task_id in tasks_db:
        #     raise HTTPException(status_code=400, detail='task in base')
        task_id = str(uuid.uuid4())
        tasks_db[task_id] = {
            "status": "RUNNING",
            "progress": 0,
            "request": req.dict(),
            "results": None,
            "error": None,
            "created_at": datetime.datetime.now(),
            # "updated_at": datetime.datetime.now(),
            "completed_at": None
        }
        background_tasks.add_task(run_analyze, task_id)
        logger.info(f"task create succes: {task_id}")
        return TaskResponse(taskid=task_id)
    except Exception as e:
        logger.error(f"task create error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get('/status')
async def get_status(task_id: str):
    if task_id not in tasks_db.keys():
        logger.error(f'task {task_id} was not detected')
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return tasks_db[task_id]['status']

@app.get('/getresult')
async def get_result(task_id: str):
    if task_id not in tasks_db.keys():
        logger.error(f'task {task_id} was not detected')
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return tasks_db[task_id]['results']


@app.get('/getresult1', response_class=HTMLResponse)
async def get_result_page(request: Request, task_id: str):
    if task_id not in tasks_db.keys():
        logger.error(f'task {task_id} was not detected')
        raise HTTPException(status_code=404, detail="Задача не найдена")

    # Получаем данные задачи
    task_data = tasks_db[task_id]

    # Передаем данные в шаблон
    return templates.TemplateResponse("results.html", {
        "request": request,
        "task_id": task_id,
        "task_data": task_data,
        "results": task_data.get('results', 'Нет результатов'),
        "status": task_data.get('status', 'UNKNOWN')
    })



# @app.get('/')
# async def index():
#     return {'hello': 'world'}
#
# @app.get('/about/')  # Исправлено здесь
# async def about(request: Request) -> MetadataResponse:
#     page = int(request.query_params.get('page', 1))
#     per_page = int(request.query_params.get('per_page', 10))
#
#     metadata = {
#         "timestamp": datetime.datetime.now().isoformat(),
#         "request_id": getattr(request.state, 'request_id', 'N/A'),
#         "endpoint": request.url.path,
#         "method": request.method,
#         "pagination": {
#             "page": page,
#             "per_page": per_page,
#             "total_items": 2,  # Это должно быть динамическим значением
#             "total_pages": 1  # Это должно быть динамическим значением
#         },
#         "version": "1.0.0"
#     }
#     '''metadata = {
#         "timestamp": datetime.now().isoformat(),
#         "request_id": getattr(request.state, 'request_id', 'N/A'),
#         "endpoint": request.url.path,
#         "method": request.method,
#         "pagination": {
#             "page": 1,
#             "per_page": 10,
#             "total_items": 2,
#             "total_pages": 1
#         },
#         "version": "1.0.0"
#     }'''
#     sample_data = ["item1", "item2"]
#     return MetadataResponse(data=sample_data, metadata=metadata)
# def f(x):
#     return x.upper()
#
# def f1():
#     g = 1
#     for i in range(10 ** 5):
#         g = (g + 1) % 2
#
# @app.get('/test/')
# async def get_score(json_data: Dict[str, Any] = Body(...)):
#     """
#     Принимает JSON в теле запроса и возвращает значение ключа 'score'
#     """
#     req_time = datetime.datetime.now()
#     ddl = json_data.get('ddl')
#     queries = json_data.get("queries")
#     url = json_data.get('url')
#     url = f(url)
#     f1()
#     resp_time = datetime.datetime.now()
#
#     # try:
#     #     ddl = json_data.get('ddl')
#     #     queries = json_data.get("queries")
#     #     url = json_data.get('url')
#         # if score is None:
#         #     raise HTTPException(status_code=400, detail="Ключ 'score' не найден в JSON")
#
#         # return {"score": str(score)}
#     return {'длл': ddl, 'куериес': queries, 'юрл': url, 'request_time': req_time,\
#             'response_time': resp_time}