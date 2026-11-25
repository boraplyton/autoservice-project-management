#!/usr/bin/env python3
"""Воспроизводимые Монте-Карло и RCPSP; только стандартная библиотека Python.

Входы: phase-4/inputs.json и phase-5/inputs.json. Все длительности RCPSP
заданы в рабочих часах, исторические трудозатраты — в человеко-часах.
"""
import argparse
import csv
import json
import math
import random
from datetime import date, datetime, time, timedelta
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def iso_week(day):
    year, week, _ = day.isocalendar()
    return f"{year}-{week:02}"


def historical_weeks(config, group_elapsed=None):
    """Задачи учитываем в неделю завершения, часы распределяем по всему периоду.

    В отчётах нет дневных записей: равномерное распределение часов внутри
    указанного периода — явное допущение, включая учебную работу в выходные.
    Пустая неделя означает отсутствие записанной работы, не доказанный простой.
    """
    first = date.fromisoformat(config["history_start"])
    last = date.fromisoformat(config["history_end"])
    if first.weekday() != 0 or last.weekday() != 6 or last >= date.fromisoformat(config["as_of"]):
        raise ValueError("История должна содержать только полные прошедшие недели")
    weeks = {}
    day = first
    while day <= last:
        weeks[iso_week(day)] = {"week": iso_week(day), "start": day.isoformat(),
                                "completed": 0, "hours": 0.0}
        day += timedelta(days=7)
    elapsed = config["phase1_group_time_is_elapsed"] if group_elapsed is None else group_elapsed
    ledger = []
    for ident, name, owner, begin, end, minutes, people, issue in config["history"]:
        start, finish = date.fromisoformat(begin), date.fromisoformat(end)
        if finish < start or minutes < 0:
            raise ValueError(f"Некорректная историческая запись {ident}")
        hours = minutes / 60 * (people if elapsed and ident.startswith("P1") else 1)
        finish_week = iso_week(finish)
        if finish_week in weeks:
            weeks[finish_week]["completed"] += 1
        days = (finish - start).days + 1
        for n in range(days):
            key = iso_week(start + timedelta(days=n))
            if key in weeks:
                weeks[key]["hours"] += hours / days
        phase = int(ident[1])
        comment = config.get("history_sources", {}).get(str(phase),
                     f"Отчёт фазы {phase}, раздел {'13' if phase == 1 else '7'}") + f"; {ident}"
        if people > 1:
            comment += "; время встречи × 3 (подтверждено пользователем)" if elapsed else "; суммарное время"
        if phase == 2:
            comment += "; трактуется как суммарные чел.-ч, общий итог отчёта 22 ч"
        url = f"https://github.com/boraplyton/autoservice-project-management/issues/{issue}" if issue else ""
        if ident == "P2-04":
            comment += "; объединены issues #19, #20, #21 в одну работу отчёта"
        ledger.append([phase, name, owner, begin, end, round(hours, 9), finish_week, url, comment])
    return list(weeks.values()), ledger


def empirical_quantile(samples, q):
    """Минимальный целый срок, эмпирическая CDF которого >= q."""
    return sorted(samples)[math.ceil(len(samples) * q) - 1]


def simulate(weekly, tasks, runs, seed):
    if not weekly or runs <= 0 or not tasks:
        raise ValueError("Нужны история, задачи и положительное число прогонов")
    counts = [w["completed"] for w in weekly]
    hours = [w["hours"] for w in weekly]
    if min(counts) < 0 or min(hours) < 0 or max(counts) <= 0 or max(hours) <= 0:
        raise ValueError("Нельзя прогнозировать по полностью нулевой/отрицательной мощности")
    for row in tasks:
        if not 0 < row[3] <= row[4] <= row[5]:
            raise ValueError(f"Нарушено 0 < a <= m <= b у {row[0]}")
    count_rng, hour_rng = random.Random(seed), random.Random(seed + 1)
    count_results, hour_results = [], []
    for _ in range(runs):
        done, week = 0, 0
        while done < len(tasks):
            done += count_rng.choice(counts)
            week += 1
        count_results.append(week)
    for _ in range(runs):
        # Одна длительность на задачу на прогон, не новый набор каждую неделю.
        remaining = sum(hour_rng.triangular(t[3], t[5], t[4]) for t in tasks)
        week = 0
        while remaining > 1e-10:
            remaining -= hour_rng.choice(hours)
            week += 1
        hour_results.append(week)
    def summary(samples):
        p80 = empirical_quantile(samples, .8)
        return {"p50_weeks": empirical_quantile(samples, .5), "p80_weeks": p80,
                "p90_weeks": empirical_quantile(samples, .9),
                "mean_weeks": sum(samples) / len(samples),
                "cdf_at_p80": sum(x <= p80 for x in samples) / len(samples),
                "cdf_before_p80": sum(x < p80 for x in samples) / len(samples)}
    return {"count": summary(count_results), "hours": summary(hour_results)}, (count_results, hour_results)


def run_phase4(config):
    folder = ROOT / "phase-4"
    if len({t[0] for t in config["meta_tasks"]}) != len(config["meta_tasks"]):
        raise ValueError("Повторный ID учебной задачи")
    for task in config["meta_tasks"]:
        if not 0 < task[3] <= task[4] <= task[5]:
            raise ValueError(f"Некорректная оценка {task[0]}")
    weeks, ledger = historical_weeks(config)
    save_csv(ROOT / "shared/Учёт_времени.csv",
             ["Фаза", "Задача", "Исполнитель", "Дата начала", "Дата завершения", "Чистое время, ч",
              "Неделя (ГГГГ-НН)", "Ссылка на Issue", "Комментарий"], ledger)
    save_csv(folder / "Недельная_статистика.csv", ["Неделя", "Понедельник", "Завершено задач", "Человеко-часы"],
             [[w["week"], w["start"], w["completed"], round(w["hours"], 6)] for w in weeks])
    save_csv(folder / "Задачи_фазы_5.csv", ["ID", "Работа", "Ответственный", "Оптимистично, чел.-ч",
             "Наиболее вероятно, чел.-ч", "Пессимистично, чел.-ч", "Предшественники"], config["meta_tasks"])
    result, _ = simulate(weeks, config["meta_tasks"], config["simulations"], config["seed"])
    result.update({"runs": config["simulations"], "seed": config["seed"], "history_weeks": len(weeks),
                   "tasks": len(config["meta_tasks"]), "forecast_start": config["forecast_start"],
                   "expected_effort_hours": sum(sum(t[3:6]) / 3 for t in config["meta_tasks"])})
    start = date.fromisoformat(config["forecast_start"])
    for key in ("count", "hours"):
        # Неделя моделируется целиком, край срока — воскресенье последней недели.
        result[key]["p80_week_end"] = (start + timedelta(weeks=result[key]["p80_weeks"], days=-1)).isoformat()
    active = [w for w in weeks if w["completed"] > 0 or w["hours"] > 0]
    result["sensitivity_active_weeks_only"] = simulate(active, config["meta_tasks"], config["simulations"], config["seed"])[0]
    for key in ("count", "hours"):
        if not result[key]["cdf_before_p80"] < .8 <= result[key]["cdf_at_p80"]:
            raise ValueError("Неверная граница P80")
    save_json(folder / "metrics.json", result)
    return result


def validate_model(config):
    tasks = {t["id"]: t for t in config["tasks"]}
    if not tasks or len(tasks) != len(config["tasks"]):
        raise ValueError("Пустой набор или повторяющиеся ID")
    caps = config["capacities"]
    if not caps or any(not isinstance(v, int) or v <= 0 for v in caps.values()):
        raise ValueError("Ресурсные мощности должны быть положительными целыми")
    for ident, task in tasks.items():
        duration = task["duration"]
        if not isinstance(duration, int) or duration <= 0:
            raise ValueError(f"Неположительная/нецелая длительность {ident}")
        if len(set(task["predecessors"])) != len(task["predecessors"]):
            raise ValueError(f"Повторные зависимости {ident}")
        if any(p not in tasks or p == ident for p in task["predecessors"]):
            raise ValueError(f"Неизвестный предшественник/петля {ident}")
        if not task["resources"] or any(k not in caps or not isinstance(v, int) or v <= 0 or v > caps[k]
                                        for k, v in task["resources"].items()):
            raise ValueError(f"Недопустимые ресурсы {ident}")
    topo, left = [], set(tasks)
    while left:
        ready = sorted(i for i in left if set(tasks[i]["predecessors"]) <= set(topo))
        if not ready:
            raise ValueError("Цикл в зависимостях")
        topo.extend(ready)
        left.difference_update(ready)
    return tasks, topo


def cpm(tasks, topo):
    successors = {i: [] for i in tasks}
    for i in tasks:
        for p in tasks[i]["predecessors"]:
            successors[p].append(i)
    es, ef, ls, lf = {}, {}, {}, {}
    for i in topo:
        es[i] = max((ef[p] for p in tasks[i]["predecessors"]), default=0)
        ef[i] = es[i] + tasks[i]["duration"]
    horizon = max(ef.values())
    for i in reversed(topo):
        lf[i] = min((ls[s] for s in successors[i]), default=horizon)
        ls[i] = lf[i] - tasks[i]["duration"]
    return {i: {"ES": es[i], "EF": ef[i], "LS": ls[i], "LF": lf[i],
                 "SLK": ls[i] - es[i],
                 "FREE": min((es[s] for s in successors[i]), default=horizon) - ef[i]}
            for i in tasks}, successors, horizon


def heuristic_orders(tasks, timing, successors, caps):
    r = lambda i: sum(tasks[i]["resources"].values())
    d = lambda i: tasks[i]["duration"]
    keys = {"SLK": lambda i: timing[i]["SLK"], "FREE": lambda i: timing[i]["FREE"],
            "LST": lambda i: timing[i]["LS"], "LFT": lambda i: timing[i]["LF"],
            "LSTLFT": lambda i: timing[i]["LS"] + timing[i]["LF"],
            "GRPW": lambda i: -(d(i) + sum(d(s) for s in successors[i])),
            "LPT": lambda i: -d(i), "MIS": lambda i: -len(successors[i]),
            "GRD": lambda i: -d(i) * r(i), "GRWC": lambda i: -r(i),
            "GCRWC": lambda i: -(r(i) + sum(r(s) for s in successors[i])),
            "ROT": lambda i: -sum(v / caps[k] for k, v in tasks[i]["resources"].items()) / d(i)}
    return {name: sorted(tasks, key=lambda i: (key(i), i)) for name, key in keys.items()}


def disjunctive_lower_bound(tasks, topo, capacities):
    """Усиливаем CPM обязательным упорядочением ресурсно несовместимых пар.

    Если две задачи вместе превышают мощность хотя бы одного ресурса,
    в любом допустимом графике одна заканчивается до начала другой.
    Минимум CPM по двум ориентациям — нижняя граница для всей задачи.
    Циклическая ориентация невозможна и исключается.
    """
    bound = cpm(tasks, topo)[2]
    certificate = None
    ids = list(tasks)
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            conflict = [k for k, cap in capacities.items()
                        if tasks[a]["resources"].get(k, 0) + tasks[b]["resources"].get(k, 0) > cap]
            if not conflict:
                continue
            alternatives = []
            for first, second in ((a, b), (b, a)):
                changed = {i: dict(t, predecessors=list(t["predecessors"])) for i, t in tasks.items()}
                if first not in changed[second]["predecessors"]:
                    changed[second]["predecessors"].append(first)
                try:
                    oriented, order = validate_model({"tasks": list(changed.values()), "capacities": capacities})
                except ValueError:
                    continue  # Цикл: это упорядочение исключено исходными связями.
                alternatives.append({"order": [first, second], "bound_hours": cpm(oriented, order)[2]})
            pair_bound = min(item["bound_hours"] for item in alternatives)
            if pair_bound > bound:
                bound = pair_bound
                certificate = {"pair": [a, b], "conflicting_resources": conflict,
                               "alternatives": alternatives, "bound_hours": pair_bound}
    return bound, certificate


def serial_schedule(tasks, order, capacities):
    """Serial SGS: приоритет среди доступных, затем самый ранний допустимый слот.

    Важны все ранее размещённые задачи, включая ещё не начавшиеся.
    При постоянных ресурсах самый ранний слот начинается в release или
    на одном из завершений уже размещённых задач: в эти моменты ресурс освобождается.
    """
    if len(order) != len(tasks) or set(order) != set(tasks):
        raise ValueError("Приоритет должен быть перестановкой всех задач")
    rank = {i: n for n, i in enumerate(order)}
    horizon = sum(t["duration"] for t in tasks.values())
    use = {k: [0] * horizon for k in capacities}
    schedule = {}
    while len(schedule) < len(tasks):
        eligible = [i for i in tasks if i not in schedule and
                    all(p in schedule for p in tasks[i]["predecessors"])]
        if not eligible:
            raise ValueError("Нет доступной задачи: проверьте DAG")
        ident = min(eligible, key=rank.get)
        task = tasks[ident]
        release = max((schedule[p][1] for p in task["predecessors"]), default=0)
        candidates = sorted({release} | {end for _, end in schedule.values() if end >= release})
        for start in candidates:
            end = start + task["duration"]
            if end <= horizon and all(all(x + need <= capacities[k] for x in use[k][start:end])
                                      for k, need in task["resources"].items()):
                schedule[ident] = (start, end)
                for k, need in task["resources"].items():
                    for hour in range(start, end):
                        use[k][hour] += need
                break
        else:
            raise ValueError(f"Не найден допустимый интервал {ident}")
    return schedule


def validate_schedule(tasks, schedule, caps):
    if set(schedule) != set(tasks):
        raise ValueError("Не все задачи размещены")
    span = max(end for _, end in schedule.values())
    use = {k: [0] * span for k in caps}
    for ident, task in tasks.items():
        start, end = schedule[ident]
        if start < 0 or end - start != task["duration"]:
            raise ValueError(f"Неверная длительность {ident}")
        if any(schedule[p][1] > start for p in task["predecessors"]):
            raise ValueError(f"Нарушение предшествования {ident}")
        for k, need in task["resources"].items():
            for hour in range(start, end):
                use[k][hour] += need
                if use[k][hour] > caps[k]:
                    raise ValueError(f"Перегрузка {k} в час {hour}")
    return use


def calendar_time(config, offset, finish=False):
    """Рабочая ось 09:00–17:00, пн–пт. Конец 8-го часа = 17:00 того же дня."""
    day_hours = config["hours_per_day"]
    if offset < 0 or (finish and offset == 0):
        raise ValueError("Некорректное смещение календаря")
    adjusted = offset - 1 if finish else offset
    day_index, hour = divmod(adjusted, day_hours)
    if finish:
        hour += 1
    day = date.fromisoformat(config["start_date"])
    if day.weekday() >= 5:
        raise ValueError("Начало проекта должно быть рабочим днём")
    while day_index:
        day += timedelta(days=1)
        if day.weekday() < 5:
            day_index -= 1
    return datetime.combine(day, time(config["workday_start_hour"])) + timedelta(hours=hour)


def assign_people(tasks, schedule, caps):
    available = {k: [0] * v for k, v in caps.items()}
    assigned = {}
    for ident in sorted(tasks, key=lambda i: (schedule[i][0], i)):
        start, end = schedule[ident]
        people = []
        for k, need in tasks[ident]["resources"].items():
            slots = [n for n, free in enumerate(available[k]) if free <= start][:need]
            if len(slots) != need:
                raise ValueError("Невозможно назначить конкретных исполнителей")
            for n in slots:
                available[k][n] = end
                people.append(f"{k}-{n + 1}")
        assigned[ident] = people
    return assigned


def gantt_svg(config, tasks, schedule, path):
    span = max(end for _, end in schedule.values())
    left, width, row, top = 490, 1020, 30, 100
    height = top + row * len(tasks) + 65
    scale = width / (math.ceil(span / 40) * 40)
    colors = {"Подготовка": "#3766a3", "Функции MVP": "#168374", "Проверка и выпуск": "#966431"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left+width+40}" height="{height}" viewBox="0 0 {left+width+40} {height}">',
           f'<rect width="{left+width+40}" height="{height}" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#18273c">',
           '<text x="24" y="32" font-size="22" font-weight="bold">ИС автосервиса: ресурсный план реализации</text>',
           '<text x="24" y="58" font-size="14">Рабочая ось: 8 ч/день, 40 ч/неделю. DEV × 2, QA × 1, BA × 1. Выходные исключены, праздники не учитываются.</text>']
    for n in range(math.ceil(span / 40) + 1):
        x = left + n * 40 * scale
        label = calendar_time(config, n * 40).strftime("%d.%m")
        out += [f'<line x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="{height-45}" stroke="#d9e1eb"/>',
                f'<text x="{x:.1f}" y="83" font-size="12">{label}</text>']
    for n, ident in enumerate(tasks):
        y = top + row * n
        task, (start, end) = tasks[ident], schedule[ident]
        if n % 2 == 0:
            out.append(f'<rect x="15" y="{y-3}" width="{left+width+10}" height="{row}" fill="#f4f7fb"/>')
        out += [f'<text x="24" y="{y+17}" font-size="13">{escape(ident + "  " + task["name"])}</text>',
                f'<rect x="{left+start*scale:.1f}" y="{y+2}" width="{(end-start)*scale:.1f}" height="21" rx="3" fill="{colors[task["stage"]]}"><title>{escape(task["name"])}: {start}–{end} рабочих ч</title></rect>']
    out += [f'<text x="24" y="{height-18}" font-size="13">Старт: {config["start_date"]}. Завершение: {calendar_time(config, span, True):%d.%m.%Y %H:%M}. Цвета: подготовка / функции MVP / проверка и выпуск.</text>', '</g></svg>']
    path.write_text("\n".join(out), encoding="utf-8")


def run_phase5(config):
    folder = ROOT / "phase-5"
    tasks, topo = validate_model(config)
    timing, successors, critical_path = cpm(tasks, topo)
    orders = heuristic_orders(tasks, timing, successors, config["capacities"])
    rng = random.Random(config["seed"])
    for n in range(config["random_orders"]):
        order = list(tasks)
        rng.shuffle(order)
        orders[f"RANDOM-{n+1:04}"] = order
    best, best_name, best_order, best_key = None, None, None, None
    comparisons = []
    for name, order in orders.items():
        schedule = serial_schedule(tasks, order, config["capacities"])
        validate_schedule(tasks, schedule, config["capacities"])
        span = max(end for _, end in schedule.values())
        key = (span, sum(end for _, end in schedule.values()))
        comparisons.append([name, span, span / 8, span / 40, key[1]])
        if best_key is None or key < best_key:
            best, best_name, best_order, best_key = schedule, name, order, key
    span = best_key[0]
    loads = {k: sum(t["duration"] * t["resources"].get(k, 0) for t in tasks.values())
             for k in config["capacities"]}
    bounds = {k: math.ceil(loads[k] / v) for k, v in config["capacities"].items()}
    disjunctive, certificate = disjunctive_lower_bound(tasks, topo, config["capacities"])
    lower = max(critical_path, disjunctive, *bounds.values())
    assigned = assign_people(tasks, best, config["capacities"])
    random_results = [row for row in comparisons if row[0].startswith("RANDOM")]
    random_best = min(random_results, key=lambda row: (row[1], row[4]))
    random_worst = max(random_results, key=lambda row: (row[1], row[4]))
    summary = [row for row in comparisons if not row[0].startswith("RANDOM")]
    summary.extend([random_best, random_worst])
    save_csv(folder / "Сравнение_эвристик.csv", ["Приоритет", "Длительность, раб.ч", "Рабочих дней", "Рабочих недель", "Сумма завершений, ч"], summary)
    save_csv(folder / "Задачи_реализации.csv", ["ID", "Задача", "Этап", "Длительность, раб.ч", "DEV", "QA", "BA", "Трудоёмкость, чел.-ч", "Предшественники", "Use Cases"],
             [[i, t["name"], t["stage"], t["duration"], *[t["resources"].get(k, 0) for k in ("DEV", "QA", "BA")],
               t["duration"] * sum(t["resources"].values()), ";".join(t["predecessors"]), ";".join(t["uc"])] for i, t in tasks.items()])
    save_csv(folder / "CPM.csv", ["ID", "EST", "EFT", "LST", "LFT", "SLK", "FREE"],
             [[i, *[timing[i][k] for k in ("ES", "EF", "LS", "LF", "SLK", "FREE")]] for i in tasks])
    save_csv(folder / "Календарный_план.csv", ["ID", "Задача", "Начало, раб.ч", "Конец, раб.ч", "Начало", "Завершение", "Исполнители (модель)"],
             [[i, tasks[i]["name"], *best[i], calendar_time(config, best[i][0]).isoformat(" "),
               calendar_time(config, best[i][1], True).isoformat(" "), ";".join(assigned[i])] for i in tasks])
    stages = []
    for stage in dict.fromkeys(t["stage"] for t in tasks.values()):
        ids = [i for i in tasks if tasks[i]["stage"] == stage]
        begin, end = min(best[i][0] for i in ids), max(best[i][1] for i in ids)
        stages.append({"name": stage, "start_hour": begin, "finish_hour": end,
                       "start": calendar_time(config, begin).isoformat(" "),
                       "finish": calendar_time(config, end, True).isoformat(" ")})
    save_csv(folder / "Этапы.csv", ["Этап", "Начало, раб.ч", "Конец, раб.ч", "Начало", "Завершение"],
             [[s[k] for k in ("name", "start_hour", "finish_hour", "start", "finish")] for s in stages])
    metrics = {"best_priority": best_name, "best_order": best_order, "makespan_hours": span,
               "working_days": span / 8, "working_weeks": span / 40,
               "finish": calendar_time(config, span, True).isoformat(" "),
               "effort_hours": sum(loads.values()), "functional_effort_hours": sum(t["duration"] * sum(t["resources"].values()) for t in tasks.values() if t["uc"]),
               "resource_effort_hours": loads, "critical_path_lower_bound_hours": critical_path,
               "resource_lower_bounds_hours": bounds, "lower_bound_hours": lower,
               "disjunctive_lower_bound_hours": disjunctive, "optimality_certificate": certificate,
               "optimal": span == lower,
               "gap_to_lower_bound_percent": (span / lower - 1) * 100,
               "utilization": {k: loads[k] / (span * cap) for k, cap in config["capacities"].items()},
               "random_orders": config["random_orders"], "seed": config["seed"],
               "random_best_hours": random_best[1],
               "random_worst_hours": random_worst[1],
               "stages": stages}
    save_json(folder / "metrics.json", metrics)
    gantt_svg(config, tasks, best, folder / "График.svg")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("4", "5", "all"), default="all")
    args = parser.parse_args()
    if args.phase in ("4", "all"):
        print(json.dumps(run_phase4(read_json(ROOT / "phase-4/inputs.json")), ensure_ascii=False, indent=2))
    if args.phase in ("5", "all"):
        print(json.dumps(run_phase5(read_json(ROOT / "phase-5/inputs.json")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
