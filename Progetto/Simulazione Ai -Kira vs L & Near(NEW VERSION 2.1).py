# === IMPORTAZIONI ===
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import random
import pandas as pd
import numpy as np
from queue import PriorityQueue
from collections import defaultdict
from pyswip import Prolog
import time

# Utility randomica
rnd = lambda a=0.0, b=1.0: random.uniform(a, b)

# === MOTORE PROLOG ===
class PrologEngine:
    def __init__(self):
        self.prolog = Prolog()
        try:
            self.prolog.consult("rules.pl")
        except Exception as e:
            print("Errore caricamento rules.pl:", e)
    def query(self, rule):
        try:
            return list(self.prolog.query(rule))
        except Exception:
            return []
    def is_liar(self,node): 
        return bool(self.query(f"mente({node})"))
    def is_buggiardo(self,node): 
        return bool(self.query(f"bugiardo({node})"))
    def is_depistatore(self,x,y): 
        return bool(self.query(f"depista({x},{y})"))
    def ha_alibi_falso(self,node): 
        return bool(self.query(f"alibi_falso({node})"))
    def è_colpevole(self,node): 
        return bool(self.query(f"colpevole({node})"))

# === MOTORE DI GIOCO TEORICO ===
class GameTheoryEngine:
    def __init__(self, agents):
        self.agents = agents
        self.payoff_matrix = {
            'Kira': {'kill': 5, 'decoy': -3, 'hide': 2},
            'Detective': {'accuse': 10, 'investigate': -2, 'wait': -5}
        }
    def nash_equilibrium(self, agent_type):
        if agent_type == 'Kira':
            return max(self.payoff_matrix['Kira'], key=lambda x: self.payoff_matrix['Kira'][x])
        else:
            return max(self.payoff_matrix['Detective'], key=lambda x: self.payoff_matrix['Detective'][x])

# === CLASSE RETE SOCIALE ===
class SocialNetwork:
    def __init__(self, num_nodes):
        self.graph = nx.DiGraph()
        self.nodes = list(range(num_nodes))
        self.graph.add_nodes_from(self.nodes)
        self.kira_node = None
        self.turn = 0
        self.victims = []
        for node in self.nodes:
            self.graph.nodes[node]['is_kira'] = False
            self.graph.nodes[node]['is_victim'] = False
            self.graph.nodes[node]['suspicion'] = 0
            self.graph.nodes[node]['interactions'] = 0
            self.graph.nodes[node]['trustworthiness'] = rnd(0.7, 1.0)
            self.graph.nodes[node]['stress'] = 0.0
            self.graph.nodes[node]['consistency'] = 1.0
    def update_emotional_state(self):
        for node in self.nodes:
            interactions = len(list(self.graph.out_edges(node)))
            self.graph.nodes[node]['stress'] = min(1.0, interactions * 0.1)
            self.graph.nodes[node]['trustworthiness'] = max(0.3, 1.0 - self.graph.nodes[node]['stress'])
    def init_kira(self, kira_node):
        self.kira_node = kira_node
        self.graph.nodes[kira_node]['is_kira'] = True
    def add_interaction(self, source, target):
        self.graph.add_edge(source, target, turn=self.turn)
        self.graph.nodes[source]['interactions'] += 1

# === KIRA ===
class Kira:
    def __init__(self, network):
        self.network = network
        self.identity = None
        self.mode = 'normal'

    def set_identity(self, node):
        self.identity = node
        self.network.init_kira(node)

    def adapt_strategy(self):
        suspicion = self.network.graph.nodes[self.identity]['suspicion']
        if suspicion > 6:
            self.mode = 'stealth'
        elif len(self.network.victims) >= 3:
            self.mode = 'chaos'
        else:
            self.mode = 'normal'

    def choose_target(self):
        candidates = [n for n in self.network.nodes if n != self.identity and not self.network.graph.nodes[n]['is_victim']]
        if not candidates:
            return None
        if self.mode == 'stealth':
            return min(candidates, key=lambda x: self.network.graph.nodes[x]['interactions'])
        elif self.mode == 'chaos':
            return max(candidates, key=lambda x: self.network.graph.nodes[x]['trustworthiness'])
        else:
            return random.choice(candidates)

    def act(self):
        self.adapt_strategy()

        # Interazione casuale (depistaggio)
        target = random.choice([n for n in self.network.nodes if n != self.identity])
        self.network.add_interaction(self.identity, target)

        # Narrativa di azione
        if 'reasons' not in self.network.graph.nodes[self.identity]:
            self.network.graph.nodes[self.identity]['reasons'] = []
        self.network.graph.nodes[self.identity]['reasons'].append(f"Kira ha agito in modalità {self.mode.upper()}")

        # Omicidio (se non stealth)
        if self.mode != 'stealth' and random.random() < 0.4:
            victim = self.choose_target()
            if victim is not None:
                self.network.graph.nodes[victim]['is_victim'] = True
                self.network.victims.append(victim)

                # Depistaggio narrativo
                if random.random() < 0.5:
                    fake_suspect = random.choice([n for n in self.network.nodes if n != self.identity and n != victim])
                    if 'reasons' not in self.network.graph.nodes[fake_suspect]:
                        self.network.graph.nodes[fake_suspect]['reasons'] = []
                    self.network.graph.nodes[fake_suspect]['reasons'].append("Indizi falsi creati da Kira")
                    self.network.graph.nodes[fake_suspect]['suspicion'] += 1


class QKira(Kira):
    def __init__(self, network):
        super().__init__(network)
        self.q_table = defaultdict(lambda: [0]*4)  # Azioni: 0=kill, 1=decoy, 2=hide, 3=wait
        self.alpha = 0.1
        self.gamma = 0.8
        self.last_action = None
        self.last_state = None

    def _get_state(self):
        # Stato semplificato: (suspicion_level, num_victims)
        suspicion = int(self.network.graph.nodes[self.identity]['suspicion'])
        num_victims = len(self.network.victims)
        return (min(suspicion, 10), min(num_victims, 10))
    def get_qtable_dataframe(self):
        rows = []
        for state, q_values in self.q_table.items():
            rows.append({
                "Stato (Susp, Vittime)": str(state),
                "Q_Kill": round(q_values[0], 2),
                "Q_Decoy": round(q_values[1], 2),
                "Q_Hide": round(q_values[2], 2),
                "Q_Wait": round(q_values[3], 2)
            })
        return pd.DataFrame(rows)


    def _calculate_reward(self):
        suspicion = self.network.graph.nodes[self.identity]['suspicion']
        fear_factor = suspicion ** 1.1  # penalità crescente
        victims = len(self.network.victims)
        return victims * 4 - fear_factor

    def choose_action(self, state):
        if random.random() < 0.2:
            return random.randint(0, 3)  # esplorazione
        return np.argmax(self.q_table[state])  # sfrutta conoscenza

    def update_q(self, state, action, reward, new_state):
        old_q = self.q_table[state][action]
        future = max(self.q_table[new_state])
        self.q_table[state][action] = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * future)

    def act(self, log=None):
        state = self._get_state()
        action = self.choose_action(state)
        action_names = ["kill", "decoy", "hide", "wait"]
        action_name = action_names[action]

        if action == 0:
            self._kill()
        elif action == 1:
            self._decoy()
        elif action == 2:
            self._hide()
        else:
            pass

        new_state = self._get_state()
        reward = self._calculate_reward()
        self.update_q(state, action, reward, new_state)

        if log is not None:
            log.append(f"QKira ha eseguito: **{action_name.upper()}**, sospetto: {self.network.graph.nodes[self.identity]['suspicion']:.1f}, vittime: {len(self.network.victims)}")




    def _kill(self):
        self.adapt_strategy()
        target = random.choice([n for n in self.network.nodes if n != self.identity])
        self.network.add_interaction(self.identity, target)

        possible_victims = [n for n in self.network.nodes if not self.network.graph.nodes[n]['is_victim'] and n != self.identity]
        if possible_victims:
            victim = random.choice(possible_victims)
            self.network.graph.nodes[victim]['is_victim'] = True
            self.network.victims.append(victim)

    def _decoy(self):
        # Depistaggio: fa sembrare qualcun altro colpevole
        others = [n for n in self.network.nodes if n != self.identity]
        if others:
            framed = random.choice(others)
            if 'reasons' not in self.network.graph.nodes[framed]:
                self.network.graph.nodes[framed]['reasons'] = []
            self.network.graph.nodes[framed]['reasons'].append("QKira ha creato indizi falsi contro questo nodo")
            self.network.graph.nodes[framed]['suspicion'] += 2

    def _hide(self):
        # Rallenta sospetto
        self.network.graph.nodes[self.identity]['suspicion'] = max(0, self.network.graph.nodes[self.identity]['suspicion'] - 2)
        if 'reasons' not in self.network.graph.nodes[self.identity]:
            self.network.graph.nodes[self.identity]['reasons'] = []
        self.network.graph.nodes[self.identity]['reasons'].append("QKira si è nascosto nell'ombra (azione hide)")


class ChaosManager:
    EVENTS = {
        'whistleblower': lambda net: net.graph.nodes[random.choice(net.nodes)].__setitem__('trustworthiness', min(1.0, net.graph.nodes[random.choice(net.nodes)]['trustworthiness'] + 0.7)),
        'mass_panic': lambda net: [net.graph.nodes[node].__setitem__('suspicion', rnd(0, 5)) for node in net.nodes],
        'blackout': lambda net: net.graph.remove_edges_from(random.sample(list(net.graph.edges), k=int(len(net.graph.edges)*0.3)))
    }
    def trigger_event(self, network):
        if random.random() < 0.15:
            event = random.choice(list(self.EVENTS.keys()))
            self.EVENTS[event](network)
            return f"**EVENTO: {event.upper()}!**"
        return None

class GeneticStrategist:
    def __init__(self, population_size=10):
        self.population = [{'aggressivity': rnd(), 'caution': rnd()} for _ in range(population_size)]
    def evolve(self, fitness_scores):
        sorted_pop = sorted(zip(self.population, fitness_scores), key=lambda x: -x[1])
        survivors = [x[0] for x in sorted_pop[:2]]
        new_pop = survivors + [self._mutate(survivors) for _ in range(8)]
        self.population = new_pop
    def _mutate(self, parent):
        return {k: v + rnd(-0.1, 0.1) for k,v in parent.items()}

# === MOTORE LOGICO DEDUTTIVO ===
class LogicalEngine:
    def __init__(self, network):
        self.network = network
        self.prolog = PrologEngine()
        self.rules = {
            'contraddizione': lambda x: f"Sospetto({x})" if self._has_contradictions(x) else None,
            'multi_vittime': lambda x: f"Colpevole({x})" if self._multiple_victim_interactions(x) else None,
            'contraddizione_incrociata': lambda x: f"Sospetto({x})" if self._cross_contradiction(x) else None
        }
    def prolog_inference(self, node):
        return self.prolog.query(f"colpevole({node})")
    def _has_contradictions(self, node):
        interactions = list(self.network.graph.out_edges(node))
        declarations = self.network.graph.nodes[node].get('declarations', [])
        return any(self._check_contradiction(i, declarations) for i in interactions)
    def _check_contradiction(self, interaction, declarations):
        return f"Interazione({interaction[0]}, {interaction[1]})" not in declarations
    def _cross_contradiction(self, node):
        declarations = self.network.graph.nodes[node].get('declarations', [])
        for other in self.network.nodes:
            if other == node: continue
            other_decls = self.network.graph.nodes[other].get('declarations', [])
            for d in declarations:
                if d in other_decls: continue
                if any(f"Interazione({other}, {node})" in other_decls for d in declarations):
                    return True
        return False
    def _multiple_victim_interactions(self, node):
        victims = [n for n in self.network.nodes if self.network.graph.nodes[n]['is_victim']]
        return len([t for s, t in self.network.graph.out_edges(node) if t in victims]) > 1
    def apply_rules(self, node):
        conclusions = []
        for rule in self.rules.values():
            result = rule(node)
            if result:
                conclusions.append(result)
        return conclusions

# === INVESTIGATION TREE ===
class InvestigationTree:
    def __init__(self, network):
        self.network = network
    def a_star_search(self, start_node):
        open_set = PriorityQueue()
        open_set.put((0, start_node))
        came_from = {}
        g_score = {node: float('inf') for node in self.network.nodes}
        g_score[start_node] = 0
        
        while not open_set.empty():
            current = open_set.get()[1]
            if self.network.graph.nodes[current]['is_kira']:
                return self.reconstruct_path(came_from, current)
            for neighbor in self.network.graph.neighbors(current):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor)
                    open_set.put((f_score, neighbor))
        return None
    def heuristic(self, node):
        return (self.network.graph.nodes[node]['suspicion'] / 10) + len(list(self.network.graph.out_edges(node)))
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

# === DETECTIVE L ===
class AdvancedDetectiveL:
    def __init__(self, network):
        self.network = network
        self.logic_engine = LogicalEngine(network)
        self.known_facts = set()
        self.actions_taken = 0

    def interrogate_node(self, node, diario, turno):
        declarations = []
        for edge in self.network.graph.out_edges(node):
            if random.random() < self.network.graph.nodes[node]['trustworthiness']:
                declarations.append(f"Interazione({edge[0]}, {edge[1]})")
            else:
                fake_target = random.choice(self.network.nodes)
                declarations.append(f"Interazione({node}, {fake_target})")
        self.network.graph.nodes[node]['declarations'] = declarations
        self.known_facts.update(declarations)

        # === PROLOG: mente(X)
        if self.logic_engine.prolog.is_liar(node):
            diario.aggiungi("L", turno, node, "Ha scoperto che ha mentito (Prolog)")
            self.network.graph.nodes[node]['suspicion'] += 3
            self._annota_reason(node, "Ha mentito (Prolog)")

        # === PROLOG: bugiardo(X)
        if self.logic_engine.prolog.is_buggiardo(node):
            diario.aggiungi("L", turno, node, "È un bugiardo recidivo (Prolog)")
            self.network.graph.nodes[node]['suspicion'] += 4
            self._annota_reason(node, "Bugiardo recidivo (Prolog)")

        # === PROLOG: depista(X,Y)
        for other in self.network.nodes:
            if other == node:
                continue
            if self.logic_engine.prolog.is_depistatore(node, other):
                diario.aggiungi("L", turno, node, f"Sta cercando di incastrare il nodo {other} (depistaggio)")
                self.network.graph.nodes[node]['suspicion'] += 2
                self._annota_reason(node, f"Ha tentato di incastrare {other} (Prolog)")

        # === PROLOG: alibi_falso(X)
        if self.logic_engine.prolog.ha_alibi_falso(node):
            diario.aggiungi("L", turno, node, "Cambia versione dei fatti (alibi falso)")
            self.network.graph.nodes[node]['suspicion'] += 3
            self._annota_reason(node, "Alibi non coerente (Prolog)")

        # === Regole standard colpevole/sospetto
        conclusions = self.logic_engine.apply_rules(node)
        for conclusion in conclusions:
            if 'Colpevole' in conclusion or 'Sospetto' in conclusion:
                suspect = int(conclusion.split('(')[1].strip(')'))
                self.network.graph.nodes[suspect]['suspicion'] += 5
                self._annota_reason(suspect, f"Detective L ha dedotto: {conclusion}")
                diario.aggiungi("L", turno, suspect, conclusion)

        return conclusions


    def _annota_reason(self, node, reason):
        if 'reasons' not in self.network.graph.nodes[node]:
            self.network.graph.nodes[node]['reasons'] = []
        self.network.graph.nodes[node]['reasons'].append(reason)

  
    def analyze_network(self, diario, turno):
        for node in self.network.nodes:
            victim_interactions = len([t for s, t in self.network.graph.out_edges(node)
                                       if self.network.graph.nodes[t]['is_victim']])
            self.network.graph.nodes[node]['suspicion'] += victim_interactions * 3
            self.interrogate_node(node, diario, turno)
    
    def make_accusation(self):
        suspicions = {n: self.network.graph.nodes[n]['suspicion'] for n in self.network.nodes}
        sorted_nodes = sorted(suspicions.items(), key=lambda item: item[1], reverse=True)
        if random.random() < 0.85:  # 85% di scegliere il più sospetto
            return sorted_nodes[0][0]
        else:  # 15% possibilità di scegliere male
            return random.choice(sorted_nodes[1:3])[0]
    


# === DETECTIVE NEAR ===
class DetectiveNear:
    def __init__(self, network):
        self.network = network
        self.actions_taken = 0

    def analyze_probabilities(self, diario=None, turno=None):
        scores = {}
        for node in self.network.nodes:
            suspicion = self.network.graph.nodes[node]['suspicion']
            interactions = len(list(self.network.graph.out_edges(node)))
            to_victims = len([t for s, t in self.network.graph.out_edges(node)
                              if self.network.graph.nodes[t]['is_victim']])
            trust = self.network.graph.nodes[node]['trustworthiness']
            stress = self.network.graph.nodes[node].get('stress', 0)
            personality = self.network.graph.nodes[node].get('personality', 'normale')

            # Punteggio dinamico
            score = (
                suspicion
                + 2 * to_victims
                + interactions * 0.3
                - trust * 4
                + stress * 1.5
            )

            # Bonus/malus psicologico
            if personality == 'paranoico':
                score += 1.5
            elif personality == 'freddo':
                score -= 1

            scores[node] = score

        sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        chosen = sorted_nodes[0][0] if random.random() < 0.8 else random.choice(sorted_nodes[1:4])[0]

        motivo = f"Interazioni: {interactions}, Vittime: {to_victims}, Stress: {stress:.2f}, Trust: {trust:.2f}"

        if diario and turno is not None:
            diario.aggiungi("Near", turno, chosen, f"Analisi statistica: {motivo}")

        if 'reasons' not in self.network.graph.nodes[chosen]:
            self.network.graph.nodes[chosen]['reasons'] = []
        self.network.graph.nodes[chosen]['reasons'].append(f"Near ha sospettato per analisi: {motivo}")

        self.network.graph.nodes[chosen]['suspicion'] += 2
        return chosen


    
# === DIARIO INVESTIGATIVO E MOTIVAZIONI ===
class DiarioInvestigativo:
    def __init__(self):
        self.entries = []

    def aggiungi(self, detective, turno, nodo, motivo):
        self.entries.append({
            "Detective": detective,
            "Turno": turno,
            "Nodo": nodo,
            "Motivo": motivo
        })

    def mostra(self):
        st.subheader("📜 Diario Investigativo Dettagliato")

        # Raggruppiamo per turno
        turni = sorted(set(entry['Turno'] for entry in self.entries))

        for turno in turni:
            with st.expander(f"🕵️‍♂️ Turno {turno}", expanded=True):
                for entry in self.entries:
                    if entry['Turno'] == turno:
                        if entry['Detective'] == "L":
                            st.markdown(
                                f"🔵 **Detective L** sospetta il nodo {entry['Nodo']} ➔ _{entry['Motivo']}_"
                            )
                        elif entry['Detective'] == "Near":
                            st.markdown(
                                f"🟣 **Detective Near** sospetta il nodo {entry['Nodo']} ➔ _{entry['Motivo']}_"
                            )



# === VISUALIZZAZIONE INTERATTIVA ===
def build_plotly_figure(network):
    G = network.graph
    pos = nx.spring_layout(G, seed=42)



    max_suspicion = max([G.nodes[node]['suspicion'] for node in G.nodes]) or 1

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_colors = []
    node_text = []
    for node in G.nodes():
        suspicion = G.nodes[node]['suspicion']
        normalized = suspicion / max_suspicion

        if normalized < 0.3:
            color = 'skyblue'
        elif normalized < 0.6:
            color = 'yellow'
        elif normalized < 0.9:
            color = 'orange'
        else:
            color = 'red'

        if node == network.kira_node:
            color = 'darkred'

        reasons_list = G.nodes[node].get('reasons', ['Nessuna motivazione registrata'])
        reasons_text = "<br>• " + "<br>• ".join(reasons_list)

        node_text.append(
            f"Nodo {node}<br>Sospetto: {suspicion:.1f}<br><br><b>Motivazioni:</b>{reasons_text}"
        )
        node_colors.append(color)


    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(color=node_colors, size=20, line_width=2)
    )

    layout = go.Layout(
        title=dict(text="🕸️ Mappa della Rete Investigativa", font=dict(size=18)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40)
    )

    return go.Figure(data=[edge_trace, node_trace], layout=layout)

class Prova:
    def __init__(self, turno, origine, descrizione):
        self.turno = turno
        self.origine = origine
        self.descrizione = descrizione

# === SIMULAZIONE STREAMLIT ===



def run_streamlit_simulation():
    st.set_page_config(page_title="Simulazione Kira vs Detective AI", layout="wide")
    # === Cutscene Narrativa Iniziale ===
    with st.container():
        st.markdown("""<h2 style='text-align: center;'>🎬 PRELUDIO</h2>""", unsafe_allow_html=True)
        placeholder = st.empty()
        narrative_lines = [
            "Una rete sociale. Un assassino nascosto tra le connessioni...",
            "I sospetti aumentano. Le vittime cadono una dopo l'altra.",
            "Solo due menti brillanti possono fermare l'orrore: L e Near.",
            "Ma Kira è scaltro. E il tempo stringe..."
        ]
        for line in narrative_lines:
            placeholder.markdown(f"<div style='font-size:18px; text-align:center;'>{line}</div>", unsafe_allow_html=True)
            time.sleep(10)
        time.sleep(10)
        placeholder.empty()

    # === Intro Cinematografico Ordinato ===
    with st.container():
        st.markdown("""
        <style>
        .title-text {
            font-size:40px;
            font-weight:bold;
            color: #FF4B4B;
            text-align: center;
        }
        .subtitle-text {
            font-size:20px;
            font-weight:normal;
            color: #333333;
            text-align: center;
        }
        .section-box {
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.05);
            color: #EEE;
            margin-bottom: 20px;
        }
        </style>
        <div class='title-text'>🔎 KIRA vs DETECTIVE AI</div>
        <div class='subtitle-text'>Benvenuto in una simulazione investigativa interattiva</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='section-box'>
        👤 **Trama**

        Una serie di misteriosi omicidi si sta diffondendo nella rete sociale. Solo una mente brillante può fermare Kira prima che sia troppo tardi.

        Prendi il controllo della situazione: osserva, analizza, deduci.
        </div>

        <div class='section-box'>
        🎮 **Modalità di gioco**

        - Kira agisce in segreto: ogni turno può colpire, nascondersi o depistare
        - Detective L e Near cercano la verità usando logica e strategia
        - Tu puoi intervenire, analizzare sospetti e presentare prove
        </div>

        <div class='section-box'>
        ⚙️ **Configurazione**
        """, unsafe_allow_html=True)

    if 'reset_sim' not in st.session_state:
        st.session_state.reset_sim = False

    if st.session_state.reset_sim:
        st.session_state.reset_sim = False

    ruolo = st.radio("🎭 Che ruolo vuoi interpretare?", ["Spettatore", "Detective L", "Detective Near"], index=0)
    st.session_state['ruolo'] = ruolo
    

    num_nodes = st.slider("🔢 Numero di nodi nella rete", 5, 30, 15)
    turni = st.slider("⏱️ Turni di gioco", 3, 20, 10)
    use_qkira = st.radio("🧠 Seleziona il tipo di Kira:", ["Kira classico", "QKira con apprendimento"], index=0)

    with st.container():
        st.markdown("""
        </div>
        <div class='section-box'>
        ✅ Quando sei pronto, clicca per iniziare la simulazione.
        </div>
        """, unsafe_allow_html=True)

    if st.button("Inizia Simulazione", key="start_simulazione"):
        st.session_state['start_sim'] = True
        st.session_state['network'] = SocialNetwork(num_nodes)
        st.session_state['detectiveL'] = AdvancedDetectiveL(st.session_state['network'])
        st.session_state['detectiveNear'] = DetectiveNear(st.session_state['network'])
        st.session_state['diario'] = DiarioInvestigativo()
        st.session_state['turno'] = 1
        network = SocialNetwork(num_nodes)

        if use_qkira == "QKira con apprendimento":
            kira = QKira(network)
            kira_log = []
        else:
            kira = Kira(network)
            kira_log = []

        kira.set_identity(random.choice(network.nodes))
        detectiveL = AdvancedDetectiveL(network)
        detectiveNear = DetectiveNear(network)
        diario = DiarioInvestigativo()
        prove_raccolte = []
        sospettati_obiettabili = set()
        punteggi = {"L": 0, "Near": 0, "Kira": 0}
        kira_suspicion_timeline = []
        vittime_timeline = []
        stress_timeline = []
        def analizza_node(node, turno, diario, prove_raccolte, prolog):
            motivazioni = []
            if prolog.is_liar(node):
                motivazioni.append("Ha mentito (mente/1)")
            if prolog.is_buggiardo(node):
                motivazioni.append("È considerato bugiardo (bugiardo/1)")
            if prolog.ha_alibi_falso(node):
                motivazioni.append("Ha cambiato alibi (alibi_falso/1)")
            if prolog.è_colpevole(node):
                motivazioni.append("Due vittime lo collegano (colpevole/1)")
            if motivazioni:
                for m in motivazioni:
                    prove_raccolte.append(Prova(turno, node, m))
                    diario.aggiungi("L", turno, node, m)

        pass
   # === Modalità giocatore attivo: Detective L interrogatorio reale ===
    if st.session_state.get('start_sim', False) and st.session_state['ruolo'] == "Detective L":
        st.title("🧠 Detective L — Interrogatorio Manuale")
        st.markdown("""
            In questa modalità, scegli tu chi interrogare. Analizza i sospetti, osserva le risposte e raccogli prove.
        """)

        net = st.session_state['network']
        detectiveL = st.session_state['detectiveL']
        diario = st.session_state['diario']
        turno = st.session_state['turno']

        nodo_scelto = st.selectbox("🔍 Scegli un nodo da interrogare", net.nodes)

        if st.button("🗯️ Interroga questo nodo"):
            result = detectiveL.interrogate_node(nodo_scelto, diario, turno)
            dichiarazioni = net.graph.nodes[nodo_scelto].get('declarations', [])
            st.markdown(f"**Dichiarazioni del Nodo {nodo_scelto}:**")
            for d in dichiarazioni:
                st.markdown(f"- {d}")
            if result:
                st.markdown(f"📌 Conclusioni logiche: {', '.join(result)}")
            else:
                st.markdown("ℹ️ Nessuna contraddizione rilevata.")

        diario.mostra()
        fig = build_plotly_figure(net)
        st.plotly_chart(fig, use_container_width=True)

    # === Modalità giocatore attivo: Detective Near ===
    if st.session_state.get('start_sim', False) and st.session_state['ruolo'] == "Detective Near":
        st.title("🧠 Detective Near — Analisi Probabilistica")
        st.markdown("""
            In questa modalità, Near userà la sua logica statistica. Puoi controllare chi viene analizzato.
        """)

        net = st.session_state['network']
        diario = st.session_state['diario']
        turno = st.session_state['turno']
        detectiveNear = st.session_state['detectiveNear']

        nodo_sospetto = detectiveNear.analyze_probabilities(diario, turno)
        st.markdown(f"📍 Nodo analizzato: **{nodo_sospetto}**")

        reasons = net.graph.nodes[nodo_sospetto].get("reasons", [])
        st.markdown("**Motivazioni raccolte:**")
        for r in reasons:
            st.markdown(f"- {r}")

        diario.mostra()
        fig = build_plotly_figure(net)
        st.plotly_chart(fig, use_container_width=True)


        # === Personalità dei nodi ===
        for node in network.nodes:
            network.graph.nodes[node]['personality'] = random.choice(['bugiardo', 'freddo', 'paranoico', 'collaborativo'])

        kira_identity = kira.identity
        grafico_id = 0

        for turno in range(1, turni + 1):
            network.turn += 1

            if isinstance(kira, QKira):
                kira.act(log=kira_log)
            else:
                kira.act()

            network.update_emotional_state()

            sospetto_attuale = network.graph.nodes[kira.identity]['suspicion']
            kira_suspicion_timeline.append({"Turno": turno, "Sospetto": sospetto_attuale})
            vittime_timeline.append({"Turno": turno, "Vittime": len(network.victims)})
            stress_medio = sum(network.graph.nodes[n]['stress'] for n in network.nodes) / len(network.nodes)
            stress_timeline.append({"Turno": turno, "Stress medio": round(stress_medio, 2)})

            st.subheader(f"🕵️‍♂️ Turno {turno}")
            st.markdown(f"🗒️ Kira ha agito in modalità **{kira.mode.upper()}**")

            for node in network.nodes:
                conclusions = detectiveL.interrogate_node(node, diario, turno)
                if conclusions:
                    st.markdown(f"**Nodo {node}**: {', '.join(conclusions)}")

                if detectiveL.logic_engine.prolog.is_liar(node):
                    sospettati_obiettabili.add(node)
                    prove_raccolte.append(Prova(turno, node, "Dichiarazione contraddittoria"))

            suspect_Near = detectiveNear.analyze_probabilities(diario, turno)

            if detectiveL.logic_engine.prolog.is_buggiardo(suspect_Near):
                prove_raccolte.append(Prova(turno, suspect_Near, "Bugia seriale rilevata da Near"))

            # === Dialoghi dinamici ===
            if turno % 3 == 0:
                sospetti = [(n, network.graph.nodes[n]['suspicion']) for n in network.nodes]
                sospetti.sort(key=lambda x: x[1], reverse=True)
                top = sospetti[0][0]
                st.markdown(f"🧠 **Near:** 'Il nodo {top} sta salendo troppo... qualcosa non torna.'")
                st.markdown(f"🕶️ **L:** 'Stiamo per inchiodarlo. Continuate così.'")

        if sospettati_obiettabili:
            st.subheader("⚖️ Obiezione!")
            nodo_obiettabile = st.selectbox("Scegli un nodo sospetto per presentare un'obiezione:", list(sospettati_obiettabili))
            prova_scelta = st.selectbox("Scegli una prova da presentare:", [f"Turno {p.turno} - Nodo {p.origine}: {p.descrizione}" for p in prove_raccolte if p.origine == nodo_obiettabile])
            if st.button("🔥 Presenta Obiezione"):
                diario.aggiungi("Utente", turno, nodo_obiettabile, f"OBIEZIONE presentata: {prova_scelta}")
                network.graph.nodes[nodo_obiettabile]['suspicion'] += 4
                st.success(f"Obiezione accettata contro Nodo {nodo_obiettabile}!")

        fig = build_plotly_figure(network)
        grafico_id += 1
        st.plotly_chart(fig, use_container_width=True, key=f"grafico_turno_{turno}_{grafico_id}")

        diario.mostra()

        st.markdown("---")

        accused_L = detectiveL.make_accusation()
        accused_Near = detectiveNear.analyze_probabilities()
        kira_scappa = random.random() < 0.1

        st.subheader("🎯 Esito Finale")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Detective L accusa:** Nodo {accused_L}")
            if accused_L == kira_identity and not kira_scappa:
                st.success("✅ Ha vinto!")
                punteggi["L"] += 10
            else:
                st.error("❌ Ha sbagliato")
                punteggi["L"] -= 5
        with col2:
            st.markdown(f"**Detective Near accusa:** Nodo {accused_Near}")
            if accused_Near == kira_identity and not kira_scappa:
                st.success("✅ Ha vinto!")
                punteggi["Near"] += 10
            else:
                st.error("❌ Ha sbagliato")
                punteggi["Near"] -= 5
        with col3:
            if kira_scappa:
                st.warning("😈 Kira è SCAPPATO!")
                punteggi["Kira"] += 7
            elif accused_L != kira_identity and accused_Near != kira_identity:
                st.markdown("😈 **Kira ha vinto!**")
                punteggi["Kira"] += 10
            else:
                st.markdown("🔎 **Detectives hanno scoperto Kira!**")

        st.markdown(f"**Identità di Kira:** Nodo {kira_identity}")

        if kira_log:
            st.subheader("📜 Log Azioni Kira")
            for entry in kira_log:
                st.markdown(f"- {entry}")

        st.subheader("📉 Evoluzione del sospetto di Kira")
        df_susp = pd.DataFrame(kira_suspicion_timeline)
        st.line_chart(df_susp.set_index("Turno"))

        st.subheader("📊 Numero di Vittime nel Tempo")
        df_vittime = pd.DataFrame(vittime_timeline)
        st.line_chart(df_vittime.set_index("Turno"))

        st.subheader("📈 Stress Medio della Rete per Turno")
        df_stress = pd.DataFrame(stress_timeline)
        st.line_chart(df_stress.set_index("Turno"))

        st.subheader("🧾 Archivio Prove")
        if prove_raccolte:
            for p in prove_raccolte:
                st.markdown(f"- Turno {p.turno}: Nodo {p.origine} → {p.descrizione}")
        else:
            st.markdown("Nessuna prova raccolta.")

        st.subheader("🏆 Classifica Finale")
        classifica = pd.DataFrame([
            {"Detective": k, "Punteggio": v} for k, v in punteggi.items()
        ]).sort_values("Punteggio", ascending=False)
        st.dataframe(classifica.reset_index(drop=True))

        st.subheader("🕸️ Stato Finale della Rete")
        fig = build_plotly_figure(network)
        st.plotly_chart(fig, use_container_width=True)
# Dopo la partita, mostra il Dossier Investigativo
        with st.expander("🗂️ Dossier Investigativo Finale", expanded=True):
            st.subheader("📋 Profilo Nodi")
            nodi_data = []
            for n in network.nodes:
                nodo = network.graph.nodes[n]
                stato = "Kira" if n == kira_identity else ("Vittima" if nodo['is_victim'] else "Innocente")
                nodi_data.append({
                    "Nodo": n,
                    "Personalità": nodo['personality'],
                    "Sospetto Finale": round(nodo['suspicion'], 2),
                    "Stato": stato
                })
            df_nodi = pd.DataFrame(nodi_data).sort_values("Sospetto Finale", ascending=False)
            st.dataframe(df_nodi.reset_index(drop=True))

            st.subheader("🧾 Prove Chiave Rilevate")
            if prove_raccolte:
                for p in prove_raccolte:
                    st.markdown(f"- Turno {p.turno}: Nodo {p.origine} → {p.descrizione}")
            else:
                st.info("Nessuna prova raccolta.")

        if st.button("🔄 Nuova Partita", key="nuova_partita_{turno}"):
            st.session_state.reset_sim = True
            st.experimental_rerun()

# === AVVIO SIMULAZIONE ===
if __name__ == "__main__":
    run_streamlit_simulation()
