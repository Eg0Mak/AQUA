from collections import namedtuple
from query_score_calculate import security_score, optimization_score, performance_score, correctness_score
import sqlparse
import json
from sqlparse.sql import Identifier, IdentifierList, Where, Comparison, Token
from sqlparse.tokens import Keyword, DML, Whitespace, Punctuation
import re


TestResult = namedtuple('TestResult', ['passed', 'reason'])

sql_keywords = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER',
        'ON', 'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'TRUE', 'FALSE',
        'ORDER', 'BY', 'GROUP', 'HAVING', 'ASC', 'DESC', 'DISTINCT', 'AS', 'CASE',
        'WHEN', 'THEN', 'ELSE', 'END', 'UNION', 'ALL', 'INTERSECT', 'EXCEPT', 'EXISTS',
        'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'CREATE', 'TABLE',
        'ALTER', 'DROP', 'INDEX', 'VIEW', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES',
        'UNIQUE', 'CHECK', 'DEFAULT', 'AUTOINCREMENT', 'LIMIT', 'OFFSET', 'WITH',
        'RECURSIVE', 'OVER', 'PARTITION', 'ROW_NUMBER', 'RANK', 'DENSE_RANK'
    }

from typing import List


def extract_clean_columns_simple(row_lst: List[str]) -> List[str]:
    """
    Очищает строку от SQL ключевых слов и возвращает список чистых названий столбцов.
    """
    columns_res = []
    for column_string in row_lst:
        # Удаляем ключевые слова ASC/DESC
        clean_string = re.sub(r'\b(ASC|DESC)\b', '', column_string, flags=re.IGNORECASE)

        # Разбиваем по запятым и очищаем каждый элемент
        columns = []
        for part in clean_string.split(','):
            part_clean = part.strip()
            if part_clean and not part_clean.upper() in ['ASC', 'DESC', 'ORDER', 'BY']:
                # Убираем возможные ASC/DESC в конце каждого элемента
                part_clean = re.sub(r'\s+(ASC|DESC)$', '', part_clean, flags=re.IGNORECASE)
                if part_clean:
                    columns.append(part_clean)
        columns_res += columns

    return columns_res

def analyze_and_markup_sql(queries_data):
    """
    Главная функция, которая принимает список запросов и возвращает их полную разметку.
    Args:
        queries_data (list): Список словарей, где каждый словарь содержит "queryid" и "query".
    Returns:
        list: Список словарей с полной разметкой для каждого запроса.
    """
    results = []
    for item in queries_data:
        query_id = item.get("queryid")
        sql_query = item.get("query")

        if not query_id or not sql_query:
            continue

        markup = markup_single_query(sql_query)
        results.append({
            "queryid": query_id,
            "query": sql_query,
            "markup": markup
        })
    return results


def markup_single_query(sql):
    """
    Выполняет структурную разметку и оценку качества одного SQL-запроса.
    """
    parsed = sqlparse.parse(sql)[0]

    # 1. Структурная разметка
    operation = parsed.get_type()
    tables = extract_tables(parsed)
    filters = extract_filters(parsed)
    parameters = [token.value.lstrip(':') for token in parsed.tokens if token.value.startswith(':')]

    # Updated extraction for group_by, order_by
    group_by_cols = extract_group_by_columns(parsed)
    order_by_cols = extract_order_by_columns(parsed)

    # 2. Критерии оценки качества
    security = evaluate_security(str(parsed))
    optimization = evaluate_optimization(str(parsed))  # Pass actual join conditions
    performance = evaluate_performance(str(parsed))
    correctness = evaluate_correctness(str(parsed))

    # Классификация типа запроса
    query_type = "Data Retrieval" if operation == 'SELECT' else "Data Modification"

    # 3. Форматирование результата
    markup = {
        "table": list(tables),
        "operation": operation,
        "filter": filters,
        "group_by": group_by_cols,
        "order_by": order_by_cols,
        "parameter": parameters,
        "type": query_type,
        "security_score": security,
        "optimization_score": optimization,
        "performance_score": performance,
        "correctness_score": correctness
    }

    return markup


# --- Функции для структурной разметки ---

def extract_tables(parsed_statement):
    """
    Извлекает имена таблиц из запроса.
    """
    tables = set()
    from_seen = False

    # Handle CTEs first to get their names
    cte_names = set()
    i = 0
    while i < len(parsed_statement.tokens):
        token = parsed_statement.tokens[i]
        if token.is_keyword and token.value.upper() == 'WITH':
            i += 1
            while i < len(parsed_statement.tokens) and not (
                    parsed_statement.tokens[i].is_keyword and parsed_statement.tokens[i].value.upper() == 'SELECT'):
                if isinstance(parsed_statement.tokens[i], Identifier):
                    cte_name = parsed_statement.tokens[i].get_real_name()
                    cte_names.add(cte_name)
                if isinstance(parsed_statement.tokens[i], sqlparse.sql.Parenthesis):
                    cte_inner_parsed = sqlparse.parse(str(parsed_statement.tokens[i].tokens[1:-1]))
                    if cte_inner_parsed:
                        tables.update(extract_tables(cte_inner_parsed[0]))
                i += 1
        i += 1

    # Now extract tables from FROM clauses, excluding CTEs
    for token in parsed_statement.tokens:
        if token.is_keyword and token.value.upper() == 'FROM':
            from_seen = True
        elif from_seen:
            if isinstance(token, Identifier):
                table_name = token.get_real_name()
                if table_name not in cte_names:
                    tables.add(table_name)
            elif isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    if isinstance(identifier, Identifier):
                        table_name = identifier.get_real_name()
                        if table_name not in cte_names:
                            tables.add(table_name)
            elif token.is_keyword and token.value.upper() in ['WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT', 'JOIN',
                                                              'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
                                                              'ON']:
                from_seen = False
    return tables


def extract_filters(parsed):
    """Извлекает условия из блока WHERE."""
    filters = []
    for token in parsed.tokens:
        if isinstance(token, Where):
            # Exclude WHERE keyword, whitespace, and semicolon
            condition_tokens = [
                str(t).strip() for t in token.tokens
                if t.ttype != Keyword.WHERE and t.ttype != Whitespace and t.value != ';'
            ]
            if condition_tokens:
                filters.append(" ".join(condition_tokens).strip())
    return filters


def extract_group_by_columns(parsed_statement):
    """
    Извлекает столбцы, используемые в GROUP BY.
    """
    group_by_cols = []
    in_group_by_clause = False
    for token in parsed_statement.tokens:
        if token.is_keyword and token.value.upper() == 'GROUP BY':
            in_group_by_clause = True
            continue
        if in_group_by_clause:
            if token.is_keyword and (token.value.upper() in ['ORDER BY', 'LIMIT', 'HAVING']):
                in_group_by_clause = False
                break
            if not token.is_whitespace and token.ttype != Punctuation.Comma and token.value != ';':
                group_by_cols.append(str(token).strip())


    columns = extract_clean_columns_simple(group_by_cols)
    return [col for col in columns if col]


def extract_order_by_columns(parsed_statement):
    """
    Извлекает столбцы, используемые в ORDER BY.
    """
    order_by_cols = []
    in_order_by_clause = False
    for token in parsed_statement.tokens:
        if token.is_keyword and token.value.upper() == 'ORDER BY':
            in_order_by_clause = True
            continue
        if in_order_by_clause:
            if token.is_keyword and (token.value.upper() == 'LIMIT'):
                in_order_by_clause = False
                break
            if not token.is_whitespace and token.ttype != Punctuation.Comma and token.value != ';':
                order_by_cols.append(str(token).strip())

    columns = extract_clean_columns_simple(order_by_cols)
    return [col for col in columns if col]


# --- Функции для оценки качества ---
def evaluate_security(query):
    """
    Проверяет наличие параметризации для защиты от SQL-инъекций
    """
    print(security_score(query))
    return ['-', '+'][security_score(query)[0]]



def evaluate_optimization(parsed):
    """Оценка оптимизации."""
    return ['-', '+'][optimization_score(parsed)[0]]


def evaluate_performance(parsed):
    """Оценка скорости (производительности)."""
    return ['-', '+'][performance_score(parsed)[0]]


def evaluate_correctness(parsed):
    """Оценка правильности написания."""
    return ['-', '+'][correctness_score(parsed)[0]]


def read_json_file(filepath):
    """
    Считывает данные из JSON файла.

    Args:
        filepath (str): Путь к JSON файлу.

    Returns:
        dict: Данные, считанные из JSON файла, или None в случае ошибки.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка: Некорректный JSON формат в файле: {filepath}")
        return None
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return None

# Пример использования:
filepath = 'flights.json'
data = read_json_file(filepath)

# --- Пример использования ---

# Представим, что вы считали ваш JSON-файл в эту переменную `data_from_snippet`
data_from_snippet = [
    {
      "queryid": "10ba3c04-0f91-4ef3-a717-c1e0d33b31bc",
      "query": "WITH MonthlyFlightCounts AS ( SELECT Origin, month(FlightDate) AS Month, COUNT(*) AS TotalFlights FROM flights.public.flights GROUP BY Origin, month(FlightDate) ORDER BY Month DESC, TotalFlights DESC ), TopAirportsByMonth AS ( SELECT Month, Origin, TotalFlights, RANK() OVER (PARTITION BY Month ORDER BY TotalFlights DESC) AS AirportRank FROM MonthlyFlightCounts ), FilteredFlights AS ( SELECT f.*, CASE WHEN f.DepTimeBlk IN ('0600-0659', '0700-0759', '0800-0859', '1600-1659', '1700-1759', '1800-1859') THEN 'Peak' ELSE 'Off-Peak' END AS TimeOfDay FROM flights.public.flights f JOIN TopAirportsByMonth t ON f.Origin = t.Origin AND month(f.FlightDate) = t.Month WHERE f.Cancelled = false AND f.Diverted = false AND t.AirportRank <= 10 ) SELECT ff.Month, Origin, TimeOfDay, COUNT(*) AS TotalFlights, ROUND(AVG(TaxiOut), 2) AS AvgTaxiOut, ROUND(AVG(DepDelay), 2) AS AvgDEPDelay, ROUND(AVG(ArrDelay), 2) AS AvgARRDelay, ROUND(CORR(TaxiOut, DepDelay), 2) AS TaxiOut_DepDelay_Correlation, ROUND(CORR(TaxiOut, ArrDelay), 2) AS TaxiOut_ArrDelay_Correlation, SUM(CASE WHEN DepDel15 = 1 THEN 1 ELSE 0 END) AS DelayedFlights, ROUND( (SUM(CASE WHEN DepDel15 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2 ) AS PercentDelayed FROM FilteredFlights ff GROUP BY ff.Month, Origin, TimeOfDay ORDER BY ff.Month DESC, Origin, TimeOfDay;",
      "runquantity": 795,
      "executiontime": 20
    },
    {
      "queryid": "8abd47c0-31cb-4ba0-891f-9bac53bbc909",
      "query": "WITH AirportDiscrepancy AS ( SELECT Origin AS Airport, OriginCityName AS AirportCity, OriginState AS AirportState, 'Origin' AS AirportRole, COUNT(*) AS TotalFlights, AVG(ActualElapsedTime - CRSElapsedTime) AS AvgDiscrepancy FROM flights.public.flights GROUP BY Origin, OriginCityName, OriginState ), RankedAirports AS ( SELECT *, RANK() OVER (PARTITION BY AirportRole ORDER BY AvgDiscrepancy DESC) AS DiscrepancyRank FROM AirportDiscrepancy ) SELECT Airport, AirportCity, AirportState, TotalFlights, ROUND(AvgDiscrepancy, 2) AS AvgDiscrepancy, ROUND(SUM(CASE WHEN DepDel15 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS PercentDelayedDepartures, ROUND(SUM(CASE WHEN Diverted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS PercentDiverted, ROUND(SUM(CASE WHEN Cancelled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS PercentCancelled, ROUND(AVG(TaxiOut), 2) AS AvgTaxiTimeMinutes FROM flights.public.flights f JOIN RankedAirports ra ON f.Origin = ra.Airport WHERE ra.DiscrepancyRank <= 20 GROUP BY Airport, AirportCity, AirportState, TotalFlights, AvgDiscrepancy, PercentDelayedDepartures, PercentDiverted, PercentCancelled, DiscrepancyRank ORDER BY DiscrepancyRank;",
      "runquantity": 490,
      "executiontime": 25
    }
]

print(data['queries'][0])
# Запускаем анализ
marked_up_data = analyze_and_markup_sql(data_from_snippet)
marked_up_data = analyze_and_markup_sql([data['queries'][0]])

# Выводим результат в красивом формате JSON
print(json.dumps(marked_up_data, indent=2, ensure_ascii=False))