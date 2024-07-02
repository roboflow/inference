import types


def create_dynamic_module(code: str, module_name: str) -> types.ModuleType:
    dynamic_module = types.ModuleType(module_name)
    exec(code, dynamic_module.__dict__)
    return dynamic_module


MY_CODE = """
SOME = 31

def function(a, b):
return a + b
"""



if __name__ == '__main__':
    module = create_dynamic_module(code=MY_CODE, module_name="dynamic_module")

    print(module.function(a=[1, 2], c=[3, 4]))

    SyntaxError