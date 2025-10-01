import sqlite3
from collections import namedtuple
import json
import psycopg2
import os
import re

# Описание теста
TestResult = namedtuple('TestResult', ['passed', 'reason'])


def security_score(query):
    """
    Проверяет наличие параметризации для защиты от SQL-инъекций
    """
    # Ищем параметры в контексте SQL, а не в строковых литералах
    query_clean = re.sub(r"'.*?'", "", query)  # Удаляем строковые литералы
    query_clean = re.sub(r".*--.*", "", query_clean)  # Удаляем комментарии

    # Проверяем наличие параметров в оставшейся части запроса
    if (':' in query_clean and re.search(r':\w+', query_clean)) or \
            ('%s' in query_clean) or \
            ('?' in query_clean):
        return TestResult(True, 'Используется параметризация')

    # Проверяем конкатенацию строк - признак уязвимости
    if ('+' in query and not re.search(r"'\s*\+\s*'", query)) or \
            ('||' in query and not re.search(r"'\s*\|\|\s*'", query)):
        return TestResult(False, 'Обнаружена конкатенация строк - возможна SQL-инъекция')

    # Если запрос полностью статический (как в вашем случае)
    if not re.search(r'\$\d+', query) and not re.search(r'@\w+', query):
        return TestResult(True, 'Статический запрос без параметров')

    return TestResult(False, 'Не используется параметризация')


def performance_score(query):
    # Проверяем наличие нормализованных условий и отношения
    tables_used = []
    keywords = {'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'ON',
                'AND', 'OR', 'SELECT', 'UPDATE', 'DELETE', 'INSERT', 'INTO', 'SET',
                'VALUES', 'BETWEEN', 'LIKE', 'IN', 'EXISTS', 'GROUP', 'ORDER', 'BY',
                'HAVING', 'LIMIT', 'OFFSET', 'UNION', 'ALL', 'DISTINCT', 'AS'}

    for word in query.upper().split():
        cleaned_word = word.strip('(),;')
        if (cleaned_word and cleaned_word not in keywords and
                not cleaned_word.isdigit() and not cleaned_word.startswith("'") and
                not cleaned_word.startswith('"') and not cleaned_word.startswith(':')):
            tables_used.append(cleaned_word)

    if len(set(tables_used)) > 1:
        return TestResult(True, '')
    else:
        return TestResult(False, 'Недостаточная нормализация')


def optimization_score(query):
    # Проверка на наличие индексов и фильтров
    query_upper = query.upper()
    if ('INDEX' in query_upper or 'WHERE' in query_upper or
            'JOIN' in query_upper or 'PRIMARY KEY' in query_upper):
        return TestResult(True, '')
    else:
        return TestResult(False, 'Нет индексации для ускорения выполнения')


def correctness_score(query):
    try:
        # Базовая проверка синтаксиса без подключения к БД
        test_query = query.upper()

        # Проверяем основные SQL ключевые слова
        required_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']
        has_required = any(keyword in test_query for keyword in required_keywords)

        if not has_required:
            return TestResult(False, 'Неверный SQL синтаксис: отсутствуют ключевые слова')

        # Проверяем базовую структуру запроса
        if 'SELECT' in test_query and 'FROM' not in test_query:
            return TestResult(False, 'Неверный SELECT запрос: отсутствует FROM')

        if 'INSERT' in test_query and 'VALUES' not in test_query and 'SELECT' not in test_query:
            return TestResult(False, 'Неверный INSERT запрос')

        if 'UPDATE' in test_query and 'SET' not in test_query:
            return TestResult(False, 'Неверный UPDATE запрос: отсутствует SET')

        if 'DELETE' in test_query and 'FROM' not in test_query:
            return TestResult(False, 'Неверный DELETE запрос')

        # Проверяем сбалансированность скобок
        if test_query.count('(') != test_query.count(')'):
            return TestResult(False, 'Несбалансированные скобки')

        # Если все проверки пройдены
        return TestResult(True, '')

    except Exception as e:
        return TestResult(False, f'Ошибка анализа: {str(e)}')


def run_tests(sql_query):
    results = {}
    tests = [
        security_score,
        performance_score,
        optimization_score,
        correctness_score
    ]

    for test_func in tests:
        result = test_func(sql_query)
        results[test_func.__name__] = result.passed

        if not result.passed:
            print(f'{test_func.__name__}: - ({result.reason})')
        else:
            print(f'{test_func.__name__}: +')

    return all(results.values())


def calculate_scores(query):
    """
    Вычисляет все scores и возвращает в нужном формате
    """
    # Запускаем все тесты
    security_test = security_score(query)
    optimization_test = optimization_score(query)
    performance_test = performance_score(query)
    correctness_test = correctness_score(query)

    # Конвертируем в символьные scores
    security_score_test = "+" if security_test.passed else "-"
    optimization_score_test = "+" if optimization_test.passed else "-"
    performance_score_test = "+" if performance_test.passed else "-"
    correctness_score_test = "+" if correctness_test.passed else "-"

    return {
        "security_score": security_score_test,
        "optimization_score": optimization_score_test,
        "performance_score": performance_score_test,
        "correctness_score": correctness_score_test
    }


def run_complete_test(sql_query, query_id):
    """
    Запускает полное тестирование и возвращает результат в нужном формате
    """
    scores = calculate_scores(sql_query)

    result = {
        "id": query_id,
        "query": sql_query.strip(),
        "security_score": scores["security_score"],
        "optimization_score": scores["optimization_score"],
        "performance_score": scores["performance_score"],
        "correctness_score": scores["correctness_score"]
    }

    return result


if __name__ == "__main__":
    with open('data/questsH.json', 'r') as f:
        data = json.load(f)
    sql_queries = data['queries']

    results = []

    for q in sql_queries:
        print(f"\nTesting query: {q['queryid']}")
        result = run_complete_test(q['query'], q['queryid'])
        results.append(result)

        # Вывод в консоль
        print(f"Security: {result['security_score']}")
        print(f"Optimization: {result['optimization_score']}")
        print(f"Performance: {result['performance_score']}")
        print(f"Correctness: {result['correctness_score']}")