import types


def test_imports():
    import spider
    from spider.simulators import dexmachina

    assert isinstance(spider.ROOT, str)
    assert isinstance(dexmachina, types.ModuleType)
