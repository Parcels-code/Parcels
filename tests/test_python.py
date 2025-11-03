from parcels._python import isinstance_noimport


def test_isinstance_noimport():
    class A:
        pass

    class B:
        pass

    a = A()
    b = B()

    assert isinstance_noimport(a, "A")
    assert not isinstance_noimport(a, "B")
    assert isinstance_noimport(b, ("A", "B"))
    assert not isinstance_noimport(b, "C")
