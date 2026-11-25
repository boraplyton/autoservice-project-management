"""Проверки математических моделей и календарных границ (без внешних пакетов)."""
import itertools
import unittest
from copy import deepcopy

from solve_phases45 import (ROOT, assign_people, calendar_time, cpm, empirical_quantile,
                            disjunctive_lower_bound, historical_weeks, read_json, serial_schedule, simulate,
                            validate_model, validate_schedule)


def task(ident, duration, predecessors=(), need=1):
    return {"id": ident, "duration": duration, "predecessors": list(predecessors), "resources": {"DEV": need}}


class ForecastTests(unittest.TestCase):
    def test_deterministic_capacity(self):
        # 5 задач при мощности 2/неделю; 5 × 2 ч при мощности 4 ч/неделю.
        tasks = [[str(n), "", "", 2, 2, 2] for n in range(5)]
        result, samples = simulate([{"completed": 2, "hours": 4}], tasks, 100, 7)
        self.assertEqual(set(samples[0]), {3})
        self.assertEqual(set(samples[1]), {3})
        self.assertEqual(result["hours"]["p80_weeks"], 3)

    def test_zero_capacity_weeks_are_sampled(self):
        rows = [{"completed": 0, "hours": 0}, {"completed": 1, "hours": 1}]
        result, samples = simulate(rows, [["a", "", "", 1, 1, 1]], 20000, 19)
        # Независимый точный результат: геометрическое распределение p=1/2,
        # P(T<=2)=.75, P(T<=3)=.875, поэтому P80=3.
        self.assertEqual(result["count"]["p80_weeks"], 3)
        self.assertEqual(result["hours"]["p80_weeks"], 3)
        self.assertGreater(max(samples[0]), 3)

    def test_zero_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate([{"completed": 0, "hours": 0}], [["a", "", "", 1, 2, 3]], 1, 1)

    def test_triangular_input_validation(self):
        with self.assertRaises(ValueError):
            simulate([{"completed": 1, "hours": 1}], [["a", "", "", 3, 2, 1]], 1, 1)

    def test_inverse_cdf(self):
        self.assertEqual(empirical_quantile([1, 2, 2, 3, 9], .8), 3)

    def test_history_and_partial_week(self):
        config = read_json(ROOT / "phase-4/inputs.json")
        weeks, ledger = historical_weeks(config)
        self.assertEqual(len(weeks), 8)
        self.assertEqual(sum(w["completed"] for w in weeks), 19)
        self.assertEqual([w["completed"] for w in weeks], [13, 0, 0, 0, 0, 0, 3, 3])
        self.assertAlmostEqual(weeks[0]["hours"], 830 / 60)
        self.assertAlmostEqual(sum(r[5] for r in ledger if r[0] == 2), 22)
        self.assertEqual(sum(row[0] in (1, 2) for row in ledger), 21)
        self.assertAlmostEqual(weeks[-1]["hours"], 59 / 6)
        self.assertNotIn("2026-48", [w["week"] for w in weeks])


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.config = {"capacities": {"DEV": 2}, "tasks": [task("A", 3), task("B", 2),
                        task("C", 4, ("A",), 2), task("D", 1, ("B", "C"))]}
        self.tasks, self.topo = validate_model(self.config)

    def test_cpm_without_resource_limits(self):
        timing, successors, bound = cpm(self.tasks, self.topo)
        self.assertEqual(bound, 8)
        self.assertEqual(timing["B"]["SLK"], 5)
        self.assertEqual(timing["B"]["FREE"], 5)
        self.assertEqual(successors["A"], ["C"])

    def test_disjunctive_bound_and_optimality_certificate(self):
        # Две независимые задачи: CPM=5, но один DEV вынуждает сумму 3+5=8.
        config = {"capacities": {"DEV": 1}, "tasks": [task("A", 3), task("B", 5)]}
        tasks, topo = validate_model(config)
        bound, certificate = disjunctive_lower_bound(tasks, topo, config["capacities"])
        self.assertEqual(bound, 8)
        self.assertEqual([a["bound_hours"] for a in certificate["alternatives"]], [8, 8])
        config = read_json(ROOT / "phase-5/inputs.json")
        tasks, topo = validate_model(config)
        bound, certificate = disjunctive_lower_bound(tasks, topo, config["capacities"])
        self.assertEqual(bound, 317)
        self.assertEqual(certificate["pair"], ["D20", "D21"])

    def test_orders_against_independent_exhaustive_schedules(self):
        # Эталон перебирает ВСЕ целочисленные начала до горизонта 10,
        # независимо от SGS. Проверяем лучший результат перебора приоритетов.
        optimum = 100
        for a, b, c, d in itertools.product(range(11), repeat=4):
            starts = {"A": a, "B": b, "C": c, "D": d}
            ends = {i: starts[i] + t["duration"] for i, t in self.tasks.items()}
            if c < ends["A"] or d < max(ends["B"], ends["C"]):
                continue
            if any(sum(t["resources"]["DEV"] for i, t in self.tasks.items()
                       if starts[i] <= h < ends[i]) > 2 for h in range(max(ends.values()))):
                continue
            optimum = min(optimum, max(ends.values()))
        found = []
        for order in itertools.permutations(self.tasks):
            schedule = serial_schedule(self.tasks, order, {"DEV": 2})
            validate_schedule(self.tasks, schedule, {"DEV": 2})
            assign_people(self.tasks, schedule, {"DEV": 2})
            found.append(max(end for _, end in schedule.values()))
        self.assertEqual(optimum, 8)
        self.assertEqual(min(found), optimum)

    def test_gap_insertion(self):
        tasks = {"A": task("A", 4, need=1), "B": task("B", 3, ("A",)), "C": task("C", 3)}
        tasks["A"]["resources"] = {"BA": 1}
        schedule = serial_schedule(tasks, ["A", "B", "C"], {"DEV": 1, "BA": 1})
        self.assertEqual(schedule["C"], (0, 3))  # Вставка до уже запланированной B.

    def test_invalid_models(self):
        cycle = deepcopy(self.config)
        cycle["tasks"][0]["predecessors"] = ["D"]
        with self.assertRaises(ValueError):
            validate_model(cycle)
        impossible = deepcopy(self.config)
        impossible["tasks"][0]["resources"]["DEV"] = 3
        with self.assertRaises(ValueError):
            validate_model(impossible)

    def test_validator_detects_bad_schedule(self):
        with self.assertRaises(ValueError):
            validate_schedule(self.tasks, {"A": (0, 3), "B": (0, 2), "C": (1, 5), "D": (5, 6)}, {"DEV": 2})

    def test_real_model_covers_all_uc(self):
        config = read_json(ROOT / "phase-5/inputs.json")
        tasks, topo = validate_model(config)
        self.assertEqual(len(tasks), 24)
        self.assertEqual(sorted(u for t in tasks.values() for u in t["uc"]), [f"UC{n:02}" for n in range(1, 13)])
        metrics = read_json(ROOT / "phase-3/metrics.json")
        self.assertEqual(sum(t["duration"] for t in tasks.values() if t["uc"]), metrics["expert_hours"])
        schedule = serial_schedule(tasks, topo, config["capacities"])
        validate_schedule(tasks, schedule, config["capacities"])

    def test_calendar_day_end_and_weekend(self):
        config = read_json(ROOT / "phase-5/inputs.json")
        self.assertEqual(str(calendar_time(config, 8, True)), "2026-11-30 17:00:00")
        self.assertEqual(str(calendar_time(config, 8)), "2026-12-01 09:00:00")
        self.assertEqual(str(calendar_time(config, 40, True)), "2026-12-04 17:00:00")
        self.assertEqual(str(calendar_time(config, 40)), "2026-12-07 09:00:00")


if __name__ == "__main__":
    unittest.main()
