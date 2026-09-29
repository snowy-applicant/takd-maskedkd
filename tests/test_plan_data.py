import torch

from takd.data import Normalizer, synthetic_dataset, train_batches
from takd.engine import learning_rate
from takd.plan import GROUPS, group_runs, run_by_name, schedule


def test_schedule_covers_all_groups_in_dependency_order():
    order = schedule([0, 1, 2])
    names = [run.name for run in order]
    assert names[0] == "teacher"
    assert len(order) == 1 + 3 * (2 + 6)
    assert len(set(names)) == len(names)
    for index, run in enumerate(order):
        if run.teacher:
            assert run.teacher in names[:index]
        assert run_by_name(run.name) == run


def test_group_definitions_match_the_plan():
    a = group_runs("A", 0)
    assert [r.name for r in a] == ["assistant_masked/seed_0", "student_A/seed_0"]
    assert a[0].masked and not a[1].masked and a[1].teacher == "assistant_masked/seed_0"
    b = group_runs("B", 0)
    assert not b[0].masked and b[1].masked
    c, d = group_runs("C", 1), group_runs("D", 1)
    assert c[0].masked and c[1].masked and not d[0].masked and not d[1].masked
    e, f = group_runs("E", 2), group_runs("F", 2)
    assert len(e) == len(f) == 1 and e[0].masked and not f[0].masked and f[0].teacher == "teacher"
    assert set(GROUPS) == set("ABCDEF")


def test_batches_are_deterministic_and_flip_only_horizontally():
    data = synthetic_dataset(torch.device("cpu"), sizes=(40, 10, 10))
    norm = Normalizer(torch.device("cpu"))
    first = [(x.clone(), y.clone()) for x, y in train_batches(data.train, 16, norm, seed=3, epoch=2)]
    second = list(train_batches(data.train, 16, norm, seed=3, epoch=2))
    other = list(train_batches(data.train, 16, norm, seed=3, epoch=3))
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(first, second))
    assert not torch.equal(first[0][1], other[0][1])
    x = data.train.images[:2]
    flipped = norm(x, torch.tensor([True, False]))
    assert torch.allclose(flipped[0], norm(x[:1])[0].flip(-1))
    assert torch.allclose(flipped[1], norm(x[1:2])[0])


def test_learning_rate_schedule_matches_reference():
    cfg = {"learning_rate": 5e-5, "min_learning_rate": 1e-6, "warmup_epochs": 5}
    assert learning_rate(cfg, 1, 100) == 1e-5
    assert learning_rate(cfg, 5, 100) == 5e-5
    assert abs(learning_rate(cfg, 6, 100) - 5e-5) < 1e-12
    assert abs(learning_rate(cfg, 100, 100) - 1e-6) < 1e-12
