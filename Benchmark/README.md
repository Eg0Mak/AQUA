# 📊 Benchmark

Для оценки качества SQL-запросов в проекте используется собственный модуль [`query_benchmark.py`](./query_benchmark.py).  
Он позволяет измерить:

1. **Валидность** (корректность синтаксиса, баланс скобок, отсутствие зарезервированных слов без кавычек).  
2. **Безопасность** (нет опасных конструкций: `UNION SELECT`, `DROP`, `EXEC`, `GRANT`, и др.).  
3. **Оптимизацию** (7 основных правил):
   - отсутствие `SELECT *`;  
   - использование `IN` вместо цепочек `OR`;  
   - запрет `NOT IN`;  
   - отсутствие дублирующихся условий;  
   - `HAVING` только с агрегатами;  
   - запрет устаревших join-ов через запятую;  
   - отсутствие лишних подзапросов в `FROM`.  

## Метрики

- `validity_score` — средний балл по проверкам валидности.  
- `optimization_score` — средний балл по оптимизациям.  
- `security_score` — средний балл по безопасным правилам.  
- `final_score` — итоговый интегральный балл (по умолчанию:  
  `0.2 * validity + 0.4 * optimization + 0.2 * security`).  

## Использование

### 1. Проверка одного запроса
```bash
python query_benchmark.py
```
или из кода:
```python
from query_benchmark import evaluate_query

query = "SELECT id, name FROM users WHERE age > 18"
result = evaluate_query(query)
print(result)
```

### 2. Сравнение пары запросов
```python
from query_benchmark import evaluate_pair

input_q = "SELECT * FROM users"
output_q = "SELECT id, name FROM users"
res = evaluate_pair(input_q, output_q)
print(res["delta"])  # улучшения по метрикам
```

### 3. Массовая оценка из JSONL
Файл должен содержать строки вида:
```json
{"input": "SELECT * FROM users", "output": "SELECT id FROM users"}
```

Запуск:
```python
from query_benchmark import evaluate_jsonl

report = evaluate_jsonl("data/queries.jsonl")
print(report)
```
