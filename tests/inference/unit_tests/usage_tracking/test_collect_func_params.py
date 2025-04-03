from inference.usage_tracking.utils import collect_func_params


def test_collect_func_params():
    # given
    def func(arg1, arg2, kwarg3, kwarg4):
        pass

    # when
    func_params = collect_func_params(
        func=func,
        args=["val1", "val2"],
        kwargs={"kwarg3": "val3", "kwarg4": "val4"},
    )

    # then
    assert func_params == {
        "arg1": "val1",
        "arg2": "val2",
        "kwarg3": "val3",
        "kwarg4": "val4",
    }


def test_collect_func_params_with_default_args():
    # given
    def func(
        arg1,
        arg2="default val2",
        arg3="default val3",
        kwarg4="default val4",
        kwarg5="default val5",
    ):
        pass

    # when
    func_params = collect_func_params(
        func=func,
        args=["val1", "val2"],
        kwargs={"kwarg4": "val4"},
    )

    # then
    assert func_params == {
        "arg1": "val1",
        "arg2": "val2",
        "arg3": "default val3",
        "kwarg4": "val4",
        "kwarg5": "default val5",
    }


def test_collect_func_params_with_args_and_catchall_kwargs():
    # given
    def func(
        arg1,
        arg2="default val2",
        arg3="default val3",
        kwarg4="default val4",
        kwarg5="default val5",
        **kwargs
    ):
        pass

    # when
    func_params = collect_func_params(
        func=func,
        args=["val1", "val2"],
        kwargs={"kwarg4": "val4", "kwarg6": "val6"},
    )

    # then
    assert func_params == {
        "arg1": "val1",
        "arg2": "val2",
        "arg3": "default val3",
        "kwarg4": "val4",
        "kwarg5": "default val5",
        "kwarg6": "val6",
        "kwargs": {"kwarg4": "val4", "kwarg6": "val6"},
    }


def test_collect_func_params_inspect_cache():
    # given
    def func(
        arg1,
        arg2="default val2",
        arg3="default val3",
        kwarg4="default val4",
        kwarg5="default val5",
        **kwargs
    ):
        pass

    # when
    func_params1 = collect_func_params(
        func=func,
        args=["val11", "val12"],
        kwargs={"kwarg4": "val14", "kwarg6": "val16"},
    )
    func_params2 = collect_func_params(
        func=func,
        args=["val21", "val22"],
        kwargs={"kwarg4": "val24", "kwarg5": "val25"},
    )

    # then
    assert func_params1 == {
        "arg1": "val11",
        "arg2": "val12",
        "arg3": "default val3",
        "kwarg4": "val14",
        "kwarg5": "default val5",
        "kwarg6": "val16",
        "kwargs": {"kwarg4": "val14", "kwarg6": "val16"},
    }
    assert func_params2 == {
        "arg1": "val21",
        "arg2": "val22",
        "arg3": "default val3",
        "kwarg4": "val24",
        "kwarg5": "val25",
        "kwargs": {"kwarg4": "val24", "kwarg5": "val25"},
    }


def test_collect_func_params_with_only_kwargs():
    # given
    def func(**kwargs):
        pass

    # when
    func_params = collect_func_params(
        func=func,
        args=[],
        kwargs={"kwarg4": "val4", "kwarg6": "val6"},
    )

    # then
    assert func_params == {
        "kwargs": {"kwarg4": "val4", "kwarg6": "val6"},
        "kwarg4": "val4",
        "kwarg6": "val6",
    }


def test_collect_func_params_with_only_kwargs_and_errorneous_args_passed():
    # given
    def func(**kwargs):
        pass

    # when
    func_params = collect_func_params(
        func=func,
        args=["val1", "val2"],
        kwargs={"kwarg4": "val4", "kwarg6": "val6"},
    )

    # then
    assert func_params == {
        "kwargs": {"kwarg4": "val4", "kwarg6": "val6"},
        "kwarg4": "val4",
        "kwarg6": "val6",
    }, "Unexpected args should be ignored rather than causing errors"
