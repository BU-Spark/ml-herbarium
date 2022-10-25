import os
from ratelimit import limits, sleep_and_retry
import openai
import ast
from tqdm import tqdm
# import rptree

@sleep_and_retry
@limits(calls=10, period=60)
def call_api(function_source):
    response = openai.Completion.create(
                    engine="code-davinci-002",
                    prompt="# Python 3.8\n\n"+function_source +"\n\n# An elaborate, high quality docstring for the above function:\n\"\"\"",
                    temperature=0,
                    max_tokens=1000,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=["\"\"\""]
                    )
    return response

USER_PATH = "~/CS549_Herbarium_Project/ml-herbarium/documentation/"
# Best Practices for API Key Safety: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
def main():
    path = os.path.expanduser(USER_PATH)
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    files = [f for f in files if f.endswith(".py") and "__" not in f and ".env" not in f and "aws" not in f]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(os.path.expanduser(USER_PATH+"docs.MD"), "w") as docs:
        docs.write("# ML-Herbarium Documentation\n\n")
        # docs.write("## File Tree\n\n")
        # rptree.
        for file in tqdm(files):
            with open(file, "r") as f:
                contents = f.read()
                module = ast.parse(contents)
                definitions = [n for n in ast.walk(module) if type(n) == ast.FunctionDef]
                if len(definitions) > 0:
                    docs.write("# "+file.split("/")[-1]+"\n\n")
                    docs.write(file+"\n\n")

                for definition in definitions:
                    function_source = ast.get_source_segment(contents, definition)
                    response = call_api(function_source)
                    docs.write("## " + definition.name + "()"+"\n\n")
                    docs.write("```"+response.choices[0].text + "```\n\n")
                    docs.write("\n\n")

if __name__ == "__main__":
    main()