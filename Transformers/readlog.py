import json

with open("logs/events.out.tfevents.1605159214.LAPTOP-SIL929TO.14444.0", "rb") as f:
    a = f.read()
v = json.loads(a)
print(v)