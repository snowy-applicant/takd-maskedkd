"""The experiment as a dependency graph of training runs.

Groups (plan section 1). "mask" means MaskedKD in that distillation stage: the
frozen teacher of the stage sees only the patches its trainee attends to most.

    A  T -> A masked,     A -> S full
    B  T -> A full,       A -> S masked
    C  T -> A masked,     A -> S masked
    D  T -> A full,       A -> S full        (plain TAKD)
    E  T -> S masked                         (direct, no assistant)
    F  T -> S full                           (direct, no assistant; the baseline)

For every seed only two assistants are trained (T -> A with and without
masking); A/C share the masked one and B/D the full one. Cost accounting still
charges each group the full cost of the assistant it uses, i.e. what running
that method on its own would cost. The fine-tuned teacher is shared by all
groups and reported separately.
"""
from dataclasses import dataclass

GROUPS = {
    "A": {"assistant": True, "assistant_masked": True, "student_masked": False},
    "B": {"assistant": True, "assistant_masked": False, "student_masked": True},
    "C": {"assistant": True, "assistant_masked": True, "student_masked": True},
    "D": {"assistant": True, "assistant_masked": False, "student_masked": False},
    "E": {"assistant": False, "assistant_masked": None, "student_masked": True},
    "F": {"assistant": False, "assistant_masked": None, "student_masked": False},
}
BASELINE = "F"


@dataclass(frozen=True)
class Run:
    name: str          # also the output sub-directory
    role: str          # "teacher" | "assistant" | "student"
    seed: int
    masked: bool       # MaskedKD in this run's distillation
    teacher: str       # name of the run whose best checkpoint is the frozen teacher ("" for the teacher)

    @property
    def stage(self):
        return {"teacher": "T", "assistant": "T->A", "student": "->S"}[self.role]


def teacher_run():
    return Run("teacher", "teacher", 0, False, "")


def assistant_run(seed, masked):
    return Run(f"assistant_{'masked' if masked else 'full'}/seed_{seed}", "assistant", seed, masked, "teacher")


def student_run(group, seed):
    spec = GROUPS[group]
    source = assistant_run(seed, spec["assistant_masked"]).name if spec["assistant"] else "teacher"
    return Run(f"student_{group}/seed_{seed}", "student", seed, spec["student_masked"], source)


def group_runs(group, seed):
    """Runs whose training cost is charged to a group (teacher fine-tuning excluded)."""
    spec = GROUPS[group]
    runs = [student_run(group, seed)]
    if spec["assistant"]:
        runs.insert(0, assistant_run(seed, spec["assistant_masked"]))
    return runs


def schedule(seeds, groups=tuple(GROUPS)):
    """Execution order: the shared teacher, then seed by seed so partial results stay usable.

    Within a seed the direct baselines run first, then each assistant followed by
    the students that depend on it.
    """
    order = [teacher_run()]
    for seed in seeds:
        wanted = [g for g in ("F", "E", "D", "B", "A", "C") if g in groups]
        for group in wanted:
            for run in group_runs(group, seed):
                if run not in order:
                    order.append(run)
    return order


def run_by_name(name):
    """Inverse of Run.name: 'teacher', 'assistant_{masked|full}/seed_N', 'student_G/seed_N'."""
    if name == "teacher":
        return teacher_run()
    head, _, tail = name.partition("/seed_")
    if not tail.isdigit():
        raise ValueError(f"Unknown run name {name!r}")
    seed = int(tail)
    if head in ("assistant_masked", "assistant_full"):
        return assistant_run(seed, head == "assistant_masked")
    if head.startswith("student_") and head[len("student_"):] in GROUPS:
        return student_run(head[len("student_"):], seed)
    raise ValueError(f"Unknown run name {name!r}")
