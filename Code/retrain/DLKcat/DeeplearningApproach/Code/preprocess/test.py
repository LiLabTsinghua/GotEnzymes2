import re
def is_float(string):
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'
    if re.match(pattern, string):
        return True
    else:
        return False

num= "124.5"
print(is_float(num))