"""
local_search.py
===============
Pesquisa local: Hill Climbing a partir da solução da pesquisa Greedy.

A atribuição dos lotes aos veículos fica fixa (vem do CSP); o Hill Climbing só
muda a ORDEM das paragens de cada veículo.

Vizinhos de uma rota:
  * trocar duas paragens de posição;
  * tirar 1, 2 ou 3 paragens seguidas e pô-las noutra posição.
Um vizinho só é aceite se for viável e tiver menos km.

Duas variantes:
  * steepest - avalia todos os vizinhos e escolhe o melhor;
  * first    - percorre os vizinhos por ordem aleatória e aceita o primeiro que melhora.
Pára quando nenhum vizinho melhora (máximo local).
"""
from __future__ import annotations

import math
import random
import time

from model import Problem, Plan, check_route


def route_cost(pb: Problem, vid: int, plan: Plan) -> float:
    """km da rota, ou infinito se for inviável."""
    r = check_route(pb, pb.vehicle[vid], pb.requests_of(plan.groups), plan.route)
    return r.distance if r.feasible else math.inf


def neighbours(route):
    """Todas as rotas vizinhas de uma rota (descrição, nova rota)."""
    n = len(route)
    for i in range(n):                                   # trocar duas paragens
        for j in range(i + 1, n):
            r = list(route)
            r[i], r[j] = r[j], r[i]
            yield f"trocar paragens {i} e {j}", r
    for size in (1, 2, 3):                               # mover um bloco de paragens
        for i in range(n - size + 1):
            for j in range(n - size + 1):
                if j != i:
                    r = list(route)
                    block = r[i:i + size]
                    del r[i:i + size]
                    r[j:j] = block
                    yield f"mover {size} paragem(ns) da posição {i} para a {j}", r


def hill_climbing(pb: Problem, sol: dict, strategy="steepest", max_iter=200, seed=0):
    """Devolve (melhor solução, custo em cada iteração, movimentos aplicados, segundos)."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    current = {v: Plan(list(p.groups), list(p.route)) for v, p in sol.items()}
    cost = {v: route_cost(pb, v, p) for v, p in current.items()}
    history, applied = [sum(cost.values())], []

    for _ in range(max_iter):
        candidates = [(v, desc, r) for v, p in current.items() for desc, r in neighbours(p.route)]
        if strategy == "first":
            rng.shuffle(candidates)
        best = None                                      # (ganho, veículo, descrição, rota, custo)
        for v, desc, r in candidates:
            c = route_cost(pb, v, Plan(current[v].groups, r))
            gain = cost[v] - c
            if gain > 1e-6 and (best is None or gain > best[0]):
                best = (gain, v, desc, r, c)
                if strategy == "first":
                    break
        if best is None:                                 # máximo local
            break
        _, v, desc, r, c = best
        current[v] = Plan(current[v].groups, r)
        cost[v] = c
        history.append(sum(cost.values()))
        applied.append(f"V{v}: {desc}")

    return current, history, applied, time.perf_counter() - t0
