from micrograd.engine import Value, backward

def test_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    backward(c)
    
    print(f"a.grad: {a.grad}")
    print(f"b.grad: {b.grad}")
    
    assert a.grad == 1.0
    assert b.grad == 1.0
    print("Test passed!")

if __name__ == "__main__":
    test_add()
