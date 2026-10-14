from langchain_core.tools import tool

@tool
def get_greet(name:str)->str:
    "Generate a docs string for the user"
    return f"Hello {name},welcome to AI world"


result=get_greet.invoke({"name":"talha"})

print(result)
print(get_greet.name)
print(get_greet.description)
print(get_greet.args)
