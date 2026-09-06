"""Small explicit arithmetic/lag/rolling grammar. Never evaluates Python code."""
import ast
import operator
import re
import numpy as np
import pandas as pd


def evaluate_formula(formula, data):
    if not data:
        raise ValueError('No formula inputs')
    index = next(iter(data.values())).index
    if any(not x.index.equals(index) for x in data.values()):
        raise ValueError('Formula inputs must share an index')
    values = dict(data)
    pattern = r'\b([A-Za-z_]\w*)_t-(\d+):t\b'
    def window(m):
        return f'window({m[1]}, {int(m[2])})'
    expression = re.sub(pattern, window, formula)
    def lag(m):
        name, lag_n = m[1], int(m[2] or 0)
        if name not in data:
            raise ValueError(f'Missing formula input: {name}')
        key = f'lag_value_{len(values)}'
        values[key] = data[name].shift(lag_n)
        return key
    expression = re.sub(r'\b([A-Za-z_]\w*)_t(?:-(\d+))?\b', lag, expression)
    binary = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul, ast.Div:operator.truediv}
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int,float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub,ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op,ast.USub) else visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left),visit(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            args = [visit(a) for a in node.args]
            if node.func.id == 'window' and len(args)==2 and isinstance(args[0],pd.Series) and isinstance(args[1],int) and args[1]>0:
                return args[0].rolling(args[1],min_periods=args[1])
            if node.func.id in ('std','mean','sum','min','max') and len(args)==1 and isinstance(args[0],pd.core.window.rolling.Rolling):
                return getattr(args[0],node.func.id)()
            if node.func.id == 'sqrt' and len(args)==1:
                return np.sqrt(args[0])
        raise ValueError(f'Unsupported formula syntax: {formula}')
    result = visit(ast.parse(expression, mode='eval'))
    if not isinstance(result,(pd.Series,int,float,np.number)):
        raise ValueError('Formula must produce a series or scalar')
    return pd.Series(result,index=index,dtype=float).replace([np.inf,-np.inf],np.nan)
