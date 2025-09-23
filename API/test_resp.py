import requests
import uuid

url = "http://127.0.0.1:8000/new"
data = {
    "url": "jdbc:postgresql://localhost/test",
    "ddl": [
        {
            "statement": "CREATE TABLE test (id INT)"
        }
    ],
    "queries": [
        {
            "queryid": str(uuid.uuid4())+'1243',
            "query": "SELECT * FROM test",
            "runquantity": 1
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())