#!/usr/bin/env python3
"""Решение задачи распределения работ Фазы 3 методом MILP.
Зависимости: scipy, numpy (pip install scipy numpy).
"""
from pathlib import Path
import argparse
import csv
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = ('Роде', 'Солопанов', 'Пухов')


def load_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def solve(width=0.30):
    tasks = load_csv(ROOT / 'phase-2' / 'Задачи_фазы_3.csv')
    pref = load_csv(ROOT / 'phase-2' / 'Предпочтения.csv')
    if [r['ID'] for r in tasks] != [r['ID'] for r in pref]:
        raise ValueError('Несовпадающие ID задач в двух таблицах')
    hours = np.array([float(r['Предварительные трудозатраты, ч']) for r in tasks])
    if not np.all(hours > 0):
        raise ValueError('Трудоёмкости должны быть положительными')

    prefs = []
    for row in pref:
        vals = [row[p] for p in PEOPLE]
        if any(not v.strip() for v in vals):
            raise ValueError(f"В {row['ID']} не заполнены оценки трёх участников.")
        nums = [int(v) for v in vals]
        if any(v < 1 or v > 5 for v in nums):
            raise ValueError(f"Баллы {row['ID']} вне шкалы 1..5")
        prefs.append(nums)

    prefs = np.asarray(prefs)
    n, k = prefs.shape
    A = lil_matrix((n + k, n * k), dtype=float)
    lb = np.array([1.0] * n + [sum(hours) / k * (1 - width)] * k)
    ub = np.array([1.0] * n + [sum(hours) / k * (1 + width)] * k)
    for i in range(n):
        for j in range(k):
            A[i, i * k + j] = 1
            A[n + j, i * k + j] = hours[i]

    result = milp(
        c=-prefs.astype(float).ravel(),
        integrality=np.ones(n * k),
        bounds=Bounds(np.zeros(n * k), np.ones(n * k)),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options={'time_limit': 30}
    )
    if result.x is None:
        raise RuntimeError(f'Нет допустимого решения: {result.message}')

    assignees = np.argmax(result.x.reshape(n, k), axis=1)
    out = ROOT / 'phase-2' / 'Распределение_итог.csv'
    with out.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ID', 'Работа', 'Часы', 'Исполнитель', 'Предпочтение'])
        for i, row in enumerate(tasks):
            writer.writerow([row['ID'], row['Работа Фазы 3'], hours[i], PEOPLE[assignees[i]], prefs[i, assignees[i]]])

    print('Цель (баллы):', sum(prefs[i, assignees[i]] for i in range(n)))
    for j, p in enumerate(PEOPLE):
        print(f'{p}: {hours[assignees == j].sum():.1f} ч')
    print('Выходной файл:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imbalance', type=float, default=0.30, help='Отклонение загрузки от среднего, 0.30 = 30%%')
    args = parser.parse_args()
    if not 0 <= args.imbalance < 1:
        parser.error('--imbalance должен быть в диапазоне [0,1)')
    solve(args.imbalance)

