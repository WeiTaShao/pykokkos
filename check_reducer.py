import cupy as cp
import pykokkos as pk


pk.set_default_space(pk.Cuda)


@pk.workunit
def sum_int(i: int, acc: pk.Acc[pk.int64], arr: pk.View1D[pk.int64]):
    acc += arr[i]


@pk.workunit
def prod_int(i: int, acc: pk.Acc[pk.int64], arr: pk.View1D[pk.int64]):
    acc *= arr[i]


@pk.workunit
def min_float(i: int, acc: pk.Acc[pk.double], arr: pk.View1D[pk.double]):
    if arr[i] < acc:
        acc = arr[i]


@pk.workunit
def max_float(i: int, acc: pk.Acc[pk.double], arr: pk.View1D[pk.double]):
    if arr[i] > acc:
        acc = arr[i]


@pk.workunit
def band_int(i: int, acc: pk.Acc[pk.int64], arr: pk.View1D[pk.int64]):
    acc &= arr[i]


@pk.workunit
def bor_int(i: int, acc: pk.Acc[pk.int64], arr: pk.View1D[pk.int64]):
    acc |= arr[i]


@pk.workunit
def land_bool(i: int, acc: pk.Acc[pk.bool], arr: pk.View1D[pk.uint8]):
    acc = acc and arr[i]


@pk.workunit
def lor_bool(i: int, acc: pk.Acc[pk.bool], arr: pk.View1D[pk.uint8]):
    acc = acc or arr[i]


@pk.workunit
def maxloc_float(i: int, acc: pk.Acc[pk.double], arr: pk.View1D[pk.double]):
    if arr[i] > acc.val:
        acc.val = arr[i]
        acc.loc = i


@pk.workunit
def minloc_float(i: int, acc: pk.Acc[pk.double], arr: pk.View1D[pk.double]):
    if arr[i] < acc.val:
        acc.val = arr[i]
        acc.loc = i


@pk.workunit
def minmax_float(i: int, acc: pk.Acc[pk.double], arr: pk.View1D[pk.double]):
    if arr[i] < acc.min_val:
        acc.min_val = arr[i]
    if arr[i] > acc.max_val:
        acc.max_val = arr[i]


@pk.workunit
def minmaxloc_float(i: int, acc: pk.Acc[pk.double], arr: pk.View1D[pk.double]):
    if arr[i] < acc.min_val:
        acc.min_val = arr[i]
        acc.min_loc = i
    if arr[i] > acc.max_val:
        acc.max_val = arr[i]
        acc.max_loc = i


def as_scalar(value):
    if hasattr(value, "item"):
        return value.item()

    return value


def as_tuple(value):
    return tuple(as_scalar(v) for v in value)


def check_scalar(name, actual, expected):
    actual = as_scalar(actual)
    expected = as_scalar(expected)

    print(f"{name:9s} pykokkos={actual} expected={expected}")
    assert actual == expected


def check_float(name, actual, expected):
    actual = float(as_scalar(actual))
    expected = float(as_scalar(expected))

    print(f"{name:9s} pykokkos={actual} expected={expected}")
    assert bool(cp.isclose(actual, expected))


def check_tuple(name, actual, expected):
    actual = as_tuple(actual)
    expected = tuple(as_scalar(v) for v in expected)

    print(f"{name:9s} pykokkos={actual} expected={expected}")
    assert actual == expected


def cuda_policy(size):
    return pk.RangePolicy(pk.ExecutionSpace.Cuda, 0, size)


def bitwise_and_all(values):
    result = int(cp.asnumpy(values)[0])
    for value in cp.asnumpy(values)[1:]:
        result &= int(value)

    return result


def bitwise_or_all(values):
    result = int(cp.asnumpy(values)[0])
    for value in cp.asnumpy(values)[1:]:
        result |= int(value)

    return result


def main():
    N = 10
    int_arr = cp.random.randint(1, 5, size=N, dtype=cp.int64)
    bit_arr = cp.random.randint(0, 16, size=N, dtype=cp.int64)
    bor_arr = cp.random.randint(0, 16, size=N, dtype=cp.int64)
    bool_and_arr = cp.random.randint(0, 2, size=N, dtype=cp.uint8)
    bool_or_arr = cp.random.randint(0, 2, size=N, dtype=cp.uint8)
    float_arr = cp.random.uniform(low=-10.0, high=10.0, size=N).astype(cp.float64)

    print("int_arr:      ", int_arr)
    print("bit_arr:      ", bit_arr)
    print("bor_arr:      ", bor_arr)
    print("bool_and_arr: ", bool_and_arr)
    print("bool_or_arr:  ", bool_or_arr)
    print("float_arr:    ", float_arr)
    print()

    check_scalar(
        "Sum",
        pk.parallel_reduce(cuda_policy(int_arr.size), sum_int, reducer=pk.Sum, arr=int_arr),
        cp.sum(int_arr),
    )
    check_scalar(
        "Prod",
        pk.parallel_reduce(
            cuda_policy(int_arr.size), prod_int, reducer=pk.Prod, arr=int_arr
        ),
        cp.prod(int_arr),
    )
    check_float(
        "Min",
        pk.parallel_reduce(
            cuda_policy(float_arr.size), min_float, reducer=pk.Min, arr=float_arr
        ),
        cp.min(float_arr),
    )
    check_float(
        "Max",
        pk.parallel_reduce(
            cuda_policy(float_arr.size), max_float, reducer=pk.Max, arr=float_arr
        ),
        cp.max(float_arr),
    )
    check_scalar(
        "BAnd",
        pk.parallel_reduce(
            cuda_policy(bit_arr.size), band_int, reducer=pk.BAnd, arr=bit_arr
        ),
        bitwise_and_all(bit_arr),
    )
    check_scalar(
        "BOr",
        pk.parallel_reduce(
            cuda_policy(bor_arr.size), bor_int, reducer=pk.BOr, arr=bor_arr
        ),
        bitwise_or_all(bor_arr),
    )
    check_scalar(
        "LAnd",
        pk.parallel_reduce(
            cuda_policy(bool_and_arr.size),
            land_bool,
            reducer=pk.LAnd,
            arr=bool_and_arr,
        ),
        bool(cp.all(bool_and_arr)),
    )
    check_scalar(
        "LOr",
        pk.parallel_reduce(
            cuda_policy(bool_or_arr.size), lor_bool, reducer=pk.LOr, arr=bool_or_arr
        ),
        bool(cp.any(bool_or_arr)),
    )

    max_loc = int(cp.argmax(float_arr).item())
    min_loc = int(cp.argmin(float_arr).item())
    check_tuple(
        "MaxLoc",
        pk.parallel_reduce(
            cuda_policy(float_arr.size),
            maxloc_float,
            reducer=pk.MaxLoc,
            arr=float_arr,
        ),
        (float(cp.max(float_arr).item()), max_loc),
    )
    check_tuple(
        "MinLoc",
        pk.parallel_reduce(
            cuda_policy(float_arr.size),
            minloc_float,
            reducer=pk.MinLoc,
            arr=float_arr,
        ),
        (float(cp.min(float_arr).item()), min_loc),
    )
    check_tuple(
        "MinMax",
        pk.parallel_reduce(
            cuda_policy(float_arr.size),
            minmax_float,
            reducer=pk.MinMax,
            arr=float_arr,
        ),
        (float(cp.min(float_arr).item()), float(cp.max(float_arr).item())),
    )
    check_tuple(
        "MinMaxLoc",
        pk.parallel_reduce(
            cuda_policy(float_arr.size),
            minmaxloc_float,
            reducer=pk.MinMaxLoc,
            arr=float_arr,
        ),
        (
            float(cp.min(float_arr).item()),
            min_loc,
            float(cp.max(float_arr).item()),
            max_loc,
        ),
    )

    print("\nAll reducer checks passed.")


if __name__ == "__main__":
    main()
