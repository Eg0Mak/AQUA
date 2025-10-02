# -*- coding: utf-8 -*-
"""
Упрощённый бенчмарк SQL:
- Валидность (синтаксис и чистота)
- 7 безопасных проверок оптимизации
- Итоговый скор
"""

import re
import json
from pathlib import Path
import sqlparse

# =====================
# Вспомогательные
# =====================

def safe_upper(s: str) -> str:
    return s.upper() if isinstance(s, str) else ""

def strip_comments(sql: str) -> str:
    s = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s



def check_security(query: str) -> dict:
    q = strip_comments(query)
    up = safe_upper(q)
    results = {}

    # 1. SQL Injection patterns
    results["no_union_select"] = "UNION SELECT" not in up
    results["no_always_true_condition"] = not bool(re.search(r"(\bor\b\s+1\s*=\s*1)", up))
    results["no_always_false_condition"] = not bool(re.search(r"(\band\b\s+1\s*=\s*0)", up))

    # 2. Ограничения по DDL/DML (опасные операции)
    results["no_drop"] = "DROP" not in up
    results["no_truncate"] = "TRUNCATE" not in up
    results["no_alter"] = "ALTER" not in up
    results["no_insert"] = "INSERT" not in up
    results["no_update"] = "UPDATE" not in up
    results["no_delete"] = "DELETE" not in up

    # 3. Проверка на лишний доступ
    results["no_grant_revoke"] = not bool(re.search(r"\b(GRANT|REVOKE)\b", up))

    # 4. Проверка на опасные функции
    dangerous_funcs = ["EXEC", "EXECUTE", "XP_CMD", "SYSOBJECTS"]
    results["no_dangerous_functions"] = not any(func in up for func in dangerous_funcs)

    return results

# =====================
# Проверки валидности
# =====================

def check_validity(query: str) -> dict:
    results = {}

    # Парсится ли
    try:
        parsed = sqlparse.parse(query or "")
        results["parsable"] = bool(parsed)
    except Exception:
        results["parsable"] = False

    # Баланс скобок
    results["balanced_parentheses"] = (query.count("(") == query.count(")"))

    # Резервированные слова без кавычек
    reserved = {"ORDER", "USER", "DATE"}
    ok_reserved = True
    m = re.search(r"SELECT\s+(.*?)\s+FROM\b", safe_upper(query), flags=re.DOTALL)
    if m:
        select_list = m.group(1)
        parts = [p.strip() for p in select_list.split(",")]
        for p in parts:
            if '"' not in p:
                if p.upper() in reserved:
                    ok_reserved = False
    results["reserved_identifiers_unquoted"] = ok_reserved

    return results

# =====================
# Проверки оптимизации
# =====================

def check_optimization(query: str) -> dict:
    q = strip_comments(query)
    up = safe_upper(q)
    results = {}

    results["no_select_star"] = not bool(re.search(r"\bSELECT\s+\*", up))
    results["no_or_instead_of_in"] = not bool(re.search(r"(\w+\s*=\s*'?\w+'?\s+OR\s+\w+\s*=\s*'?\w+'?)+", up))
    results["no_not_in"] = "NOT IN" not in up

    conds = re.findall(r"(\b\w+\s*=\s*'?\w+'?\b)", up)
    results["no_duplicate_conditions"] = (len(conds) == len(set(conds)))

    if "HAVING" in up and not re.search(r"(SUM|COUNT|AVG|MIN|MAX)\s*\(", up):
        results["no_having_without_agg"] = False
    else:
        results["no_having_without_agg"] = True

    results["no_old_style_join"] = not bool(re.search(r"\bFROM\s+\w+\s*,\s*\w+", up))
    results["no_redundant_subquery"] = not bool(re.search(r"\bFROM\s*\(\s*SELECT\s+[^)]+\)", up))

    return results

# =====================
# Итоговая оценка
# =====================



def evaluate_query(query: str, alpha=0.2, beta=0.4, gamma=0.2) -> dict:
    validity = check_validity(query)
    optimization = check_optimization(query)
    security = check_security(query)

    v_score = sum(validity.values()) / len(validity) if validity else 1.0
    o_score = sum(optimization.values()) / len(optimization) if optimization else 1.0
    s_score = sum(security.values()) / len(security) if security else 1.0

    final_score = alpha * v_score + beta * o_score + gamma * s_score

    #final_score = alpha * v_score + (1 - alpha) * o_score

    return {
        "validity": validity,
        "optimization": optimization,
        "validity_score": v_score,
        "optimization_score": o_score,
        "security_score": s_score,
        "final_score": final_score
    }

# =====================
# Для пар input-output
# =====================

def evaluate_pair(input_query: str, output_query: str, alpha: float = 0.5) -> dict:
    res_in = evaluate_query(input_query, alpha)
    res_out = evaluate_query(output_query, alpha)
    return {
        "input": res_in,
        "output": res_out,
        "delta": {
            "validity": res_out["validity_score"] - res_in["validity_score"],
            "optimization": res_out["optimization_score"] - res_in["optimization_score"],
            "final": res_out["final_score"] - res_in["final_score"]
        }
    }

# =====================
# Для JSONL файлов
# =====================

def evaluate_jsonl(path: str, alpha: float = 0.5, limit: int = None) -> dict:
    path = Path(path)
    count = 0
    agg = {"input_validity": 0, "input_opt": 0, "input_final": 0,
           "output_validity": 0, "output_opt": 0, "output_final": 0}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q_in, q_out = obj.get("input", ""), obj.get("output", "")
            if not (isinstance(q_in, str) and isinstance(q_out, str)):
                continue
            r_in = evaluate_query(q_in, alpha)
            r_out = evaluate_query(q_out, alpha)
            agg["input_validity"] += r_in["validity_score"]
            agg["input_opt"] += r_in["optimization_score"]
            agg["input_final"] += r_in["final_score"]
            agg["output_validity"] += r_out["validity_score"]
            agg["output_opt"] += r_out["optimization_score"]
            agg["output_final"] += r_out["final_score"]
            count += 1
            if limit and count >= limit:
                break

    for k in agg:
        agg[k] /= max(count, 1)

    return {"count": count, "averages": agg}

if __name__ == '__main__':
    filepath = 'data/queries_marked_602_915_utf8_prep.jsonl'

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            single_query_res = evaluate_query(obj['input'])
            print(single_query_res)