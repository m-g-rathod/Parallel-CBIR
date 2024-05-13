def f(dictionary):
    dummy_dict = {}
    dummy_dict['x'] = 3
    return dummy_dict

def func(dict1):
    d1 = f(dict1)
    dict1 = d1
    return "hello", dict1