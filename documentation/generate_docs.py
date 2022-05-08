import importlib
import os
import openai
import ast
import inspect
import sys

path = os.path.expanduser("~/ml-herbarium/")
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
files = [f for f in files if f.endswith(".py")]

for file in files:
    with open(file, "r") as f:
        contents = f.read()
        module = ast.parse(contents)
        definitions = [n for n in ast.walk(module) if type(n) == ast.FunctionDef]
        # spec=importlib.util.spec_from_file_location(os.path.basename(file).split(".")[0],file)
        # foo = importlib.util.module_from_spec(spec)
        try:
            sys.path.append(os.path.dirname(file))
            __import__(os.path.basename(file).split(".")[0])
            sys.path.pop()
        except:
            continue

        for definition in definitions:
            func = getattr(os.path.basename(file).split(".")[0], definition.name)
            function_text = inspect.getsource(func)
            print(function_text)





openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-002",
  prompt="class Log:\n    def __init__(self, path):\n        dirname = os.path.dirname(path)\n        os.makedirs(dirname, exist_ok=True)\n        f = open(path, \"a+\")\n\n        # Check that the file is newline-terminated\n        size = os.path.getsize(path)\n        if size > 0:\n            f.seek(size - 1)\n            end = f.read(1)\n            if end != \"\\n\":\n                f.write(\"\\n\")\n        self.f = f\n        self.path = path\n\n    def log(self, event):\n        event[\"_event_id\"] = str(uuid.uuid4())\n        json.dump(event, self.f)\n        self.f.write(\"\\n\")\n\n    def state(self):\n        state = {\"complete\": set(), \"last\": None}\n        for line in open(self.path):\n            event = json.loads(line)\n            if event[\"type\"] == \"submit\" and event[\"success\"]:\n                state[\"complete\"].add(event[\"id\"])\n                state[\"last\"] = event\n        return state\n\n\"\"\"\nHere's what the above class is doing:\n1.",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\"\"\""]
)