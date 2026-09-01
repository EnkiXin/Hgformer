import pytest
import torch

from recbole.evaluator.collector import Collector, DataStruct


def test_data_struct_accumulates_multiple_batches_equivalently(monkeypatch):
    batches = [
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        torch.tensor([[5, 6]], dtype=torch.int64),
        torch.tensor([[7, 8], [9, 10]], dtype=torch.int64),
    ]
    expected = torch.cat(batches, dim=0)
    original_cat = torch.cat
    cat_calls = 0

    def counted_cat(*args, **kwargs):
        nonlocal cat_calls
        cat_calls += 1
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", counted_cat)
    data = DataStruct()

    for batch in batches:
        data.update_tensor("rec.topk", batch)

    assert cat_calls == 0
    assert torch.equal(data.get("rec.topk"), expected)
    assert cat_calls == 1


def test_data_struct_empty_and_single_batch_cases():
    empty = DataStruct()
    empty.finalize_tensors()
    assert "rec.topk" not in empty

    zero_rows = DataStruct()
    zero_rows.update_tensor("rec.topk", torch.empty((0, 3), dtype=torch.float32))
    assert zero_rows["rec.topk"].shape == (0, 3)

    source = torch.tensor([[1.5, 2.5]], dtype=torch.float64, requires_grad=True)
    single = DataStruct()
    single.update_tensor("rec.score", source)
    source.data.zero_()
    result = single["rec.score"]
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert not result.requires_grad
    assert torch.equal(result, torch.tensor([[1.5, 2.5]], dtype=torch.float64))


def test_materialized_value_can_receive_another_batch():
    data = DataStruct()
    data.update_tensor("rec.score", torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.equal(data["rec.score"], torch.tensor([[1.0]]))

    data.update_tensor("rec.score", torch.tensor([[2.0]], dtype=torch.float32))

    assert torch.equal(data["rec.score"], torch.tensor([[1.0], [2.0]]))


def test_collector_finalizes_and_resets_batched_resources():
    collector = Collector.__new__(Collector)
    collector.data_struct = DataStruct()
    collector.data_struct.set("data.num_items", 4)
    collector.data_struct.update_tensor(
        "rec.topk", torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    )
    collector.data_struct.update_tensor(
        "rec.topk", torch.tensor([[1, 1]], dtype=torch.int32)
    )

    returned = collector.get_data_struct()

    assert returned["rec.topk"].dtype == torch.int32
    assert torch.equal(
        returned["rec.topk"],
        torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32),
    )
    assert returned["data.num_items"] == 4
    assert "rec.topk" not in collector.data_struct
    assert collector.data_struct["data.num_items"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_batches_are_collected_on_cpu_with_dtype_preserved():
    data = DataStruct()
    data.update_tensor(
        "rec.score", torch.tensor([[1.0, 2.0]], device="cuda", dtype=torch.float16)
    )
    data.update_tensor(
        "rec.score", torch.tensor([[3.0, 4.0]], device="cuda", dtype=torch.float16)
    )

    result = data["rec.score"]
    assert result.device.type == "cpu"
    assert result.dtype == torch.float16
    assert torch.equal(
        result, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    )
