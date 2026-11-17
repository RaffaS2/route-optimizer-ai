"""
csp.py
======
Atribuição dos lotes de recolha aos veículos, formulada como um CSP
(Constraint Satisfaction Problem) e resolvida com backtracking,
forward checking e AC-3.

Formulação
----------
Variáveis : X_g, uma por lote de recolha g = (produtor, dia).
Domínio   : D(X_g) = {veículos}  (no tiny e no small: {1, 2}).
Restrições duras:
  C1 (unária)  o lote cabe no veículo e consegue ser servido sozinho por ele;
  C2 (binária) dois lotes que, juntos, não cabem numa rota viável do veículo v
               não podem ficar ambos em v:  NOT (X_g = v AND X_h = v);
  C3 (global)  todos os lotes de um veículo têm de caber numa rota que respeite
               janelas de recolha e de entrega, capacidade e passagem pelo armazém;
  C4 (global)  ... e nenhum produto fica mais de 24h no armazém.
  C2-C4 são verificadas com o verificador de rotas + pesquisa gulosa
  (informed.route_search), que com a poda por janelas é completa.

Porque é que C4 é tratada à parte?
  C1-C3 são "monótonas": se um conjunto de lotes é inviável, juntar-lhe mais
  lotes nunca o torna viável. Por isso podemos usá-las para podar cedo
  (consistência, forward checking e AC-3). A regra das 24h não é monótona:
  mais lotes podem atrasar a chegada ao armazém e assim ajudar a cumpri-la.
  Por isso C4 só é verificada quando a atribuição está completa.
Restrições suaves (preferências), usadas para ordenar os valores a testar:
  S1 não dividir encomendas entre veículos (uma só entrega por encomenda);
  S2 percorrer menos km.
Heurística de escolha de variável: MRV e, em empate, o grau.
"""
from __future__ import annotations

import itertools
import time

from model import Problem
from informed import route_search


class RoutingCSP:
    def __init__(self, pb: Problem, max_expanded=20_000):
        self.pb = pb
        self.max_expanded = max_expanded
        self.variables = list(pb.groups)                    # ordem cronológica
        self.values = [v.vehicle_id for v in pb.vehicles]
        self._routes = {}                                   # cache das pesquisas de rota
        self.checks = 0

        # pedidos -> encomendas de cada lote (para a restrição suave S1)
        self.orders_of = {g: {pb.requests[r].order_id for r in pb.groups[g]} for g in self.variables}

        # C1: consistência de nó (restrição unária)
        self.domains = {g: [v for v in self.values if self.feasible(v, [g])]
                        for g in self.variables}

        # C2: pares de lotes incompatíveis no mesmo veículo (pré-calculado)
        self.incompatible = set()
        for g, h in itertools.combinations(self.variables, 2):
            for v in self.values:
                if not self.feasible(v, [g, h]):
                    self.incompatible.add((g, h, v))
                    self.incompatible.add((h, g, v))
        self.neighbors = {g: {h for (a, h, _) in self.incompatible if a == g} for g in self.variables}

    # --- verificação das restrições ---------------------------------------------
    def route(self, vid, groups, dwell=False):
        """Rota encontrada pela pesquisa gulosa para estes lotes (com cache)."""
        key = (vid, frozenset(groups), dwell)
        if key not in self._routes:
            reqs = self.pb.requests_of(groups)
            self._routes[key] = route_search(self.pb, self.pb.vehicle[vid], reqs,
                                             "greedy", "h2_mst", self.max_expanded, dwell)
        return self._routes[key]

    def feasible(self, vid, groups, dwell=False) -> bool:
        """Existe rota viável para estes lotes? dwell=False -> C1-C3 (para podar);
        dwell=True -> C1-C4 (verificação final)."""
        self.checks += 1
        return self.route(vid, groups, dwell).route is not None

    def compatible(self, g, x, h, y) -> bool:
        """Restrição binária C2 entre X_g = x e X_h = y."""
        return not (x == y and (g, h, x) in self.incompatible)

    # --- AC-3 --------------------------------------------------------------------
    def ac3(self, domains, queue=None) -> bool:
        """Torna os arcos consistentes; devolve False se algum domínio ficar vazio."""
        if queue is None:
            queue = [(g, h) for g in self.variables for h in self.neighbors[g]]
        while queue:
            g, h = queue.pop()
            if self._revise(domains, g, h):
                if not domains[g]:
                    return False
                queue.extend((k, g) for k in self.neighbors[g] if k != h)
        return True

    def _revise(self, domains, g, h) -> bool:
        removed = False
        for x in list(domains[g]):
            if not any(self.compatible(g, x, h, y) for y in domains[h]):
                domains[g].remove(x)
                removed = True
        return removed

    # --- heurísticas -------------------------------------------------------------
    def select_variable(self, assignment, domains, heuristics):
        unassigned = [g for g in self.variables if g not in assignment]
        if not heuristics:
            return unassigned[0]                             # ordem fixa (cronológica)
        # MRV: o lote com menos veículos possíveis; em empate, o grau: o lote com
        # mais restrições com lotes ainda por atribuir
        return min(unassigned, key=lambda g: (
            len(domains[g]), -sum(1 for h in self.neighbors[g] if h not in assignment)))

    def order_values(self, g, assignment, domains, heuristics):
        if not heuristics:
            return list(domains[g])

        def score(v):
            mine = [h for h, x in assignment.items() if x == v]
            others = {o for h, x in assignment.items() if x != v for o in self.orders_of[h]}
            split = len(self.orders_of[g] & others)                          # S1: encomendas divididas
            added = self.route(v, mine + [g]).cost - (self.route(v, mine).cost if mine else 0)  # S2: km a mais
            return (split, added)
        return sorted(domains[g], key=score)

    # --- backtracking ------------------------------------------------------------
    def solve(self, forward_checking=True, ac3=True, heuristics=True):
        """Devolve (atribuição {vehicle_id: [lotes]} ou None, estatísticas)."""
        self._routes.clear()                                 # tempos comparáveis entre variantes
        t0, checks0 = time.perf_counter(), self.checks
        self.backtracks = self.nodes = 0
        domains = {g: list(d) for g, d in self.domains.items()}
        result = None
        if not ac3 or self.ac3(domains):
            result = self._backtrack({}, domains, forward_checking, ac3, heuristics)
        stats = {"dataset": self.pb.name, "forward checking": forward_checking, "AC-3": ac3,
                 "heurísticas": heuristics, "solução": result is not None,
                 "atribuições testadas": self.nodes, "backtracks": self.backtracks,
                 "verificações de restrições": self.checks - checks0,
                 "tempo (s)": round(time.perf_counter() - t0, 3)}
        if result is None:
            return None, stats
        return {v: [g for g in self.variables if result[g] == v] for v in self.values}, stats

    def _backtrack(self, assignment, domains, fc, ac3, heuristics):
        if len(assignment) == len(self.variables):           # atribuição completa
            for v in self.values:                            # verificar C4 (24h no armazém)
                if not self.feasible(v, [g for g, x in assignment.items() if x == v], dwell=True):
                    return None
            return dict(assignment)
        g = self.select_variable(assignment, domains, heuristics)
        for v in self.order_values(g, assignment, domains, heuristics):
            self.nodes += 1
            mine = [h for h, x in assignment.items() if x == v]
            if not self.feasible(v, mine + [g]):             # C2 + C3 com as atribuições feitas
                continue
            assignment[g] = v
            new = {h: list(d) for h, d in domains.items()}
            new[g] = [v]
            ok = True
            if fc:                                           # forward checking
                for h in self.variables:
                    if h not in assignment and v in new[h] and not self.feasible(v, mine + [g, h]):
                        new[h].remove(v)
                        if not new[h]:
                            ok = False
                            break
            if ok and ac3:                                   # manter consistência de arco (MAC)
                ok = self.ac3(new, [(h, g) for h in self.neighbors[g] if h not in assignment])
            if ok:
                result = self._backtrack(assignment, new, fc, ac3, heuristics)
                if result is not None:
                    return result
            del assignment[g]                                # desfazer: backtrack
            self.backtracks += 1
        return None
