import asyncio
from fastapi import Request, FastAPI, BackgroundTasks, HTTPException
from schemas import TaskResponse, OptimizationRequest, StatusResponse
import datetime
import uuid
import logging
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch
from sentence_transformers import SentenceTransformer
import pickle
from transformers import AutoTokenizer
import gc
from typing import Dict, Any, List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
from fastapi.templating import Jinja2Templates
torch.cuda.empty_cache()
templates = Jinja2Templates(directory="templates")
TIMEOUT_SECONDS = 20 * 60
tasks_db = {}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
charts_data = {}
model, model_sql, index, metas, texts, tokenizer = None, None, None, None, None, None
def get_gpu_memory():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }
    return {}

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info(f"Memory after cleanup: {get_gpu_memory()}")

def _update_task_progress_and_status(task_id: str):
    task = tasks_db.get(task_id)
    if not task:
        return
    qmap = task.get("queries", {})
    total = len(qmap)
    results = task.get("results")
    if not isinstance(results, dict):
        results = {
            "queries": {},
            "ddl": task.get("request", {}).get("ddl"),
            "dll": task.get("request", {}).get("ddl"),
            "migrations": None
        }

    if total == 0:
        task["progress"] = 100
        results["queries"] = {}
        task["results"] = results
        tasks_db[task_id] = task
        return
    queries_results = {qid: v.get("results", "") for qid, v in qmap.items()}
    results["queries"] = queries_results
    task["results"] = results
    done_count = sum(1 for v in qmap.values() if v.get("status") == "DONE")
    try:
        task["progress"] = int((done_count / total) * 100)
    except ZeroDivisionError:
        task["progress"] = 0
    statuses = {v.get("status") for v in qmap.values()}
    if all(s == "DONE" for s in statuses):
        task["status"] = "DONE"
        dates = [v.get("completed_at") for v in qmap.values() if v.get("completed_at")]
        task["completed_at"] = max(dates) if dates else datetime.datetime.now()
    elif any(s == "RUNNING" for s in statuses):
        task["status"] = "RUNNING"
    else:
        task["status"] = "FAILED"
        task["completed_at"] = datetime.datetime.now()

    tasks_db[task_id] = task

def load_model():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index("RAG2/faiss_index.idx")
    with open("RAG2/texts_metas.pkl", "rb") as f:
        data = pickle.load(f)
    texts = data["texts"]
    metas = data["metas"]
    tokenizer = AutoTokenizer.from_pretrained('sqlcoder_finetuned_2')
    merged_model_path = 'sqlcoder_finetuned_2'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model_sql = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        device_map="cuda",
        quantization_config=bnb_config
    )
    return model, model_sql, index, metas, texts, tokenizer


def search(query, model, index, texts, metas, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    return [
        {"text": texts[i], "metadata": metas[i], "distance": float(distances[0][j])}
        for j, i in enumerate(indices[0])
    ]


def blocking_generate(prompt, tokenizer, model_sql, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model_sql.generate(
            **inputs,
            max_new_tokens=100,  # Максимальное количество генерируемых токенов
            do_sample=True,  # Включаем сэмплинг (вместо жадного выбора)
            temperature=0.7,  # "Температура" для сэмплинга
            top_k=30,  # Ограничиваем выбор 50 самыми вероятными токенами
            top_p=0.9,  # Нуклеарный сэмплинг с порогом 0.9
            num_beams=5,  # Количество лучей в beam search
            repetition_penalty=2.5,  # Штраф за повторения токенов
            early_stopping=True,  # Останавливать генерацию при EOS токене
            num_return_sequences=1  # Количество возвращаемых вариантов генерации
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split("Optimized SQL:")[-1].strip()


async def generate_optimized_sql_few_shot(query, retrieved_docs):
    context_text = """
"""
    if retrieved_docs:
        context_text += "\n\nAdditional context:\n" + "\n".join([doc['text'] for doc in retrieved_docs])


    few_shot_examples = """
    Example 1:
    Original SQL:
    SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE status='active');
    Optimized SQL:
    SELECT o.*
    FROM orders o
    JOIN users u ON o.user_id = u.id
    WHERE u.status='active';

    Example 2:
    Original SQL:
    SELECT u.id, (SELECT COUNT(*) FROM orders o WHERE o.user_id=u.id) AS order_count FROM users u;
    Optimized SQL:
    SELECT u.id, COUNT(o.id) AS order_count
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    GROUP BY u.id;

    Example 3:
    Original SQL:
    SELECT o.id, o.total, u.name
    FROM orders o
    JOIN users u ON o.user_id = u.id
    WHERE u.status='active' AND o.total > 100;
    Optimized SQL:
    SELECT o.id, o.total, u.name
    FROM orders o
    JOIN users u ON o.user_id = u.id
    WHERE u.status='active' AND o.total > 100;
    -- Этот запрос уже оптимален, изменений не требуется

    Example 4:
    Original SQL:
    WITH recent_orders AS (
        SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '30 days'
    )
    SELECT u.id, u.name, COUNT(r.id)
    FROM users u
    JOIN recent_orders r ON u.id = r.user_id
    GROUP BY u.id, u.name;
    Optimized SQL:
    SELECT u.id, u.name, COUNT(o.id)
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE o.created_at > NOW() - INTERVAL '30 days'
    GROUP BY u.id, u.name;

    Example 5:
    Original SQL:
    SELECT u.id, u.name,
          (SELECT MAX(p.created_at) FROM payments p WHERE p.user_id=u.id) AS last_payment
    FROM users u;
    Optimized SQL:
    SELECT u.id, u.name, MAX(p.created_at) AS last_payment
    FROM users u
    LEFT JOIN payments p ON u.id = p.user_id
    GROUP BY u.id, u.name;

    Example 6:
    Original SQL:
    WITH recent_orders AS (
        SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '30 days'
    ),
    completed_payments AS (
        SELECT * FROM payments WHERE status = 'completed'
    )
    SELECT u.id, u.name, r.id AS order_id, r.total, p.id AS payment_id, p.created_at AS payment_date
    FROM users u
    JOIN recent_orders r ON u.id = r.user_id
    LEFT JOIN completed_payments p ON r.id = p.order_id
    WHERE u.status = 'active'
    ORDER BY r.total DESC;
    Optimized SQL:
    SELECT u.id, u.name, o.id AS order_id, o.total, p.id AS payment_id, p.created_at AS payment_date
    FROM users u
    JOIN orders o
        ON u.id = o.user_id
        AND o.created_at > NOW() - INTERVAL '30 days'
    LEFT JOIN payments p
        ON o.id = p.order_id
        AND p.status = 'completed'
    WHERE u.status = 'active'
    ORDER BY o.total DESC;

    Example 7:
    Original SQL:
    SELECT u.id, u.name, o.id AS order_id, o.total,
          SUM(p.amount) OVER(PARTITION BY u.id ORDER BY p.created_at) AS cumulative_payments
    FROM users u
    JOIN orders o ON u.id = o.user_id
    LEFT JOIN payments p ON o.id = p.order_id
    WHERE u.status='active';
    Optimized SQL:
    SELECT u.id, u.name, o.id AS order_id, o.total,
          SUM(p.amount) OVER(PARTITION BY u.id ORDER BY p.created_at) AS cumulative_payments
    FROM users u
    JOIN orders o ON u.id = o.user_id
    LEFT JOIN payments p ON o.id = p.order_id AND p.status='completed'
    WHERE u.status='active';
    -- Добавлен фильтр p.status='completed' для оптимизации
    """


    prompt = f"""
I am a senior SQL developer with many years of experience, deeply understanding the philosophy of writing SQL code, its subtle points, and potential vulnerabilities. Please optimize the following SQL query as efficiently and safely as possible. Full context of the database and environment is provided.

Tasks:
1. Analyze the query in terms of performance: estimate number of operations, iterations, and volume of data read.
2. Identify bottlenecks and potential vulnerabilities, including SQL injection risks, suboptimal joins, unnecessary subqueries, and aggregations.
3. Apply best practices for query optimization where possible, such as: proper indexing, reordering of filter conditions (WHERE, ON), optimizing join order, minimizing data volume in intermediate steps, using window functions or CTEs.
4. Provide a revised query that is fully detailed, safe, and includes an explanation of each change. Preserve the semantic meaning of the original query. If a JOIN can be safely removed without changing the semantics, do so.
5. Take into account the specific database system, version, and configuration. For repeated computations, suggest caching or materialization strategies.
6. Emphasize security: eliminate any risks of data leakage or unintended operations.
7. Additionally, create a concise operations map with estimated time and computational complexity. Where possible, provide alternative optimization options and tools for monitoring and diagnostics.

For analysis, I provide the original SQL query, the execution plan, and database structure (if necessary).

Context:
{context_text}

{few_shot_examples}

Now optimize this SQL query:

Original SQL:
{query}

Optimized SQL:
"""


    result = await run_in_threadpool(
        blocking_generate,
        prompt,
        tokenizer,
        model_sql,
        app.state.device
    )
    return result



@app.on_event("startup")
async def startup_event():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.device = device
    logger.info(f"Using device: {device}")
    logger.info(f"Initial GPU memory: {get_gpu_memory()}")
    try:
        logger.info("[startup] Loading models...")
        global model, model_sql, index, metas, texts, tokenizer
        model, model_sql, index, metas, texts, tokenizer = await run_in_threadpool(load_model)
        logger.info('state current')
    except Exception as e:
        logger.exception("Failed to load resources: %s", e)
        print("model load error:", e)
    try:
        logger.info("Warming up model_sql...")
        f = open('warm_model.txt')
        warm_prompt = f.read()
        f.close()
        await run_in_threadpool(
            blocking_generate,
            warm_prompt,
            tokenizer,
            model_sql,
            app.state.device
        )
        logger.info("Warm-up done")
    except Exception as e:
        logger.exception("Warm-up failed: %s", e)
    MAX_CONCURRENT_BG_TASKS = int(os.environ.get("MAX_CONCURRENT_BG_TASKS", 3))
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENT_BG_TASKS)
async def run_analyze(task_id: str, query: str = None, queryid: str = None):
    sem = app.state.semaphore
    if task_id not in tasks_db:
        logger.error("run_analyze called for unknown task_id %s", task_id)
        return
    task_entry = tasks_db[task_id]
    q_entry = task_entry["queries"][queryid]
    async def _do_work():
        try:
            q_entry["status"] = "RUNNING"
            q_entry["started_at"] = datetime.datetime.now()
            q_entry["error"] = None
            q_entry["results"] = None
            q_entry["completed_at"] = None

            # Поиск релевантных документов (если есть модель и индекс)
            retrieved = []
            try:
                if query and model is not None and index is not None:
                    retrieved = await run_in_threadpool(
                        search, query, model, index, texts, metas, 5
                    )
            except Exception:
                logger.exception("Error during search for task %s query %s", task_id, queryid)

            # Генерация результата (async)
            try:
                result = await generate_optimized_sql_few_shot(query, retrieved)
                q_entry["status"] = "DONE"
                q_entry["results"] = result
                q_entry["error"] = None
                logger.info("Subtask %s:%s DONE", task_id, queryid)
            except Exception:
                logger.exception("Error during generate_optimized_sql for subtask %s:%s", task_id, queryid)
                q_entry["status"] = "FAILED"
                q_entry["results"] = ""
                q_entry["error"] = "generation_error"

            q_entry["completed_at"] = datetime.datetime.now()

            # Обновляем агрегированные поля задачи НЕ создавая новые ключи
            # (функция _update_task_progress_and_status использует существующую структуру)
            try:
                _update_task_progress_and_status(task_id)
            except Exception:
                logger.exception("Failed to update task progress/status for %s after subtask %s", task_id, queryid)

        except Exception as e:
            logger.exception("Unexpected error in _do_work for %s:%s -> %s", task_id, queryid, e)
            # аккуратно помечаем подзадачу как FAILED (без setdefault)
            try:
                q_entry["status"] = "FAILED"
                q_entry["results"] = ""
                q_entry["error"] = str(e)
                q_entry["completed_at"] = datetime.datetime.now()
                _update_task_progress_and_status(task_id)
            except Exception:
                logger.exception("Also failed to mark subtask failed for %s:%s", task_id, queryid)

    try:
        async with sem:
            await asyncio.wait_for(_do_work(), timeout=TIMEOUT_SECONDS)

    except asyncio.TimeoutError:
        logger.warning("Subtask %s:%s timed out after %s seconds", task_id, queryid, TIMEOUT_SECONDS)
        try:
            q_entry["status"] = "FAILED"
            q_entry["results"] = ""
            q_entry["error"] = "timeout"
            q_entry["completed_at"] = datetime.datetime.now()
            _update_task_progress_and_status(task_id)
        except Exception:
            logger.exception("Failed to mark timeout for subtask %s:%s", task_id, queryid)
    except Exception as e:
        logger.exception("Unexpected error in run_analyze for %s:%s -> %s", task_id, queryid, e)
        try:
            q_entry["status"] = "FAILED"
            q_entry["results"] = ""
            q_entry["error"] = str(e)
            q_entry["completed_at"] = datetime.datetime.now()
            _update_task_progress_and_status(task_id)
        except Exception:
            logger.exception("Also failed to mark unexpected error for %s:%s", task_id, queryid)


@app.get('/')
async def base():
    return tasks_db


@app.get("/health")
async def health():
    return {"ok": True, "device": app.state.device}


@app.post("/new", response_model=TaskResponse)
async def create_optimization_task(req: OptimizationRequest, background_tasks: BackgroundTasks):
    try:
        if not req.ddl:
            raise HTTPException(status_code=400, detail="DDL not found")
        if not req.queries:
            raise HTTPException(status_code=400, detail="Queries not found")

        task_id = str(uuid.uuid4())
        now = datetime.datetime.now()

        # Инициализируем структуру задачи
        tasks_db[task_id] = {
            "status": "RUNNING",
            "progress": 0,
            "request": req.dict(),
            "results": {"queries": {}, "ddl": req.ddl, "migrations": []},
            "error": None,
            "created_at": now,
            "completed_at": None,
            "queries": {}
        }

        for q in req.queries:
            logger.info(q, type(q))
            qid = q.queryid
            qtext = q.query
            tasks_db[task_id]["queries"][qid] = {
                "query": qtext,
                "status": "PENDING",
                "results": None,
                "error": None,
                "created_at": now,
                "started_at": None,
                "completed_at": None
            }
            background_tasks.add_task(run_analyze, task_id, qtext, qid)
            logger.info("Scheduled background subtask for task %s queryid %s", task_id, qid)

        logger.info(f"task create success: {task_id}")
        return TaskResponse(taskid=task_id)

    except Exception as e:
        logger.error(f"task create error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get('/status')
async def get_status(task_id: str):
    if task_id not in tasks_db.keys():
        logger.error(f'task {task_id} was not detected')
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return {"status": tasks_db[task_id]['status']}


@app.get('/getresult')
async def get_result(task_id: str):
    if task_id not in tasks_db.keys():
        logger.error(f'task {task_id} was not detected')
        raise HTTPException(status_code=404, detail="Задача не найдена")
    a = tasks_db[task_id]['results']
    qurs = a['queries']
    vals = list(qurs.values())
    keys = list(qurs.keys())
    a['queries'] = []
    for i in range(len(keys)):
        c = {'queryid': vals[i], 'query': keys[i]}
        a['queries'].append(c)
    return tasks_db[task_id]['results']

# ТЕСТ ВЫВОДА БЕНЧМАРКОВ
# @app.get('/getresult2', response_class=HTMLResponse)
# async def get_result_page(request: Request, task_id: str):
#     if task_id not in tasks_db.keys():
#         logger.error(f'task {task_id} was not detected')
#         raise HTTPException(status_code=404, detail="Задача не найдена")
#
#     task_data = tasks_db[task_id]
#
#     return templates.TemplateResponse("results2.html", {
#         "request": request,
#         "task_id": task_id,
#         "task_data": task_data,
#         "results": task_data.get('results', 'Нет результатов'),
#         "status": task_data.get('status', 'UNKNOWN'),
#         'charts': charts_data
#     })