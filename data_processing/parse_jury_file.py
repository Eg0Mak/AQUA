import json

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

if __name__ == "__main__":
    data = read_json_file('data/flights.json')
    print(data['ddl'])