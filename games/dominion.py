from __future__ import annotations

import datetime
import math
import pathlib
import random
from typing import Dict, List, Optional, Set

import numpy as np

from .abstract_game import AbstractGame
from .dominion_base import (
    Game as DominionGame,
    CardType,
    TurnPhase,
    BASE_2E_KINGDOM_CARDS,
    ALL_CARDS,
)

# ----------------- STABLE ACTION SPACE (BY CARD NAME) -----------------

N_CARDS = len(ALL_CARDS)

# 0..N_CARDS-1: select a card by name (meaning depends on phase)
ACTION_SELECT_OFFSET = 0

# N_CARDS..2*N_CARDS-1: select a supply pile by name (buy/gain phases)
ACTION_SUPPLY_OFFSET = N_CARDS

# Fixed small choice actions
CHOICE_0 = 2 * N_CARDS + 0
CHOICE_1 = 2 * N_CARDS + 1
CHOICE_2 = 2 * N_CARDS + 2

ACTION_END_ACTION = 2 * N_CARDS + 3
ACTION_END_TURN = 2 * N_CARDS + 4

ACTION_SPACE_SIZE = 2 * N_CARDS + 5

# ----------------- Reward shaping config -----------------
# 0.25 * Δ(VP_diff) за шаг, небольшой штраф за ошибку и за длину партии.
#REWARD_VP_DIFF_SCALE = 0.25      # шаговый шейпинг: 0.25 * (VP_diff_after - VP_diff_before)
REWARD_ILLEGAL_ACTION = -0.1     # небольшой штраф за нелегальное действие
REWARD_STEP_PENALTY = -0.0005     # штраф за "затягивание" партии на каждый шаг

# Potential-based VP shaping
REWARD_SHAPING_NORM   = 6.0    # 6 VP = Province
REWARD_SHAPING_ALPHA  = 0.1    # alpha, о которой мы говорили

# ----------------- MuZero-like config -----------------
class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None

        self.observation_shape = (1, 1, 170)  # set by adapter
        self.action_space = list(range(ACTION_SPACE_SIZE))

        # true 2-player env
        self.players = [0, 1]
        self.stacked_observations = 0

        self.muzero_player = 0
        self.opponent = "self"

        self.num_workers = 4
        self.selfplay_on_gpu = False
        self.max_moves = 600
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.network = "resnet"
        self.support_size = 15

        self.downsample = False
        self.blocks = 2
        self.channels = 32
        self.reduced_channels_reward = 32
        self.reduced_channels_value = 32
        self.reduced_channels_policy = 32
        self.resnet_fc_reward_layers = [32]
        self.resnet_fc_value_layers = [32]
        self.resnet_fc_policy_layers = [32]

        self.encoding_size = 32
        self.fc_representation_layers = [32]
        self.fc_dynamics_layers = [32]
        self.fc_reward_layers = [32]
        self.fc_value_layers = [32]
        self.fc_policy_layers = [32]

        self.results_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )
        self.save_model = True
        self.training_steps = 500_000
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = True

        self.optimizer = "SGD"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.03
        self.lr_decay_rate = 0.75
        self.lr_decay_steps = 500_000

        self.replay_buffer_size = 10000
        self.num_unroll_steps = 50
        self.td_steps = 100
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None

    def next_to_play(self, current_to_play: int, action: int) -> int:
        """
        Решаем, какой логический игрок ходит в следующем узле дерева.

        Правила:
        - По умолчанию, пока ход не завершён, решения принимает тот же игрок.
        - При END_TURN ход передаётся следующему игроку по кругу.
        - При розыгрыше Militia / Bureaucrat в ACTION-фазе
          следующая интерактивная фаза принадлежит жертве,
          поэтому в виртуале переключаемся на оппонента.

        Это приближённая модель (мы не знаем фактическое состояние среды
        в MCTS), но роли атакующего и жертвы по крайней мере честно
        разделены.
        """

        players = getattr(self, "players", None)
        if not players:
            return current_to_play

        # 1) Конец хода: передаём ход следующему игроку по кругу.
        if action == ACTION_END_TURN:
            try:
                idx = players.index(current_to_play)
            except ValueError:
                idx = current_to_play
            return players[(idx + 1) % len(players)]

        # 2) Розыгрыш атакующей карты, дающей решения оппоненту:
        #    Militia / Bureaucrat.
        if ACTION_SELECT_OFFSET <= action < ACTION_SELECT_OFFSET + N_CARDS:
            card_name = ALL_CARDS[action - ACTION_SELECT_OFFSET]
            if card_name in ("Militia", "Bureaucrat"):
                try:
                    idx = players.index(current_to_play)
                except ValueError:
                    idx = current_to_play
                # передаём управление оппоненту (жертве)
                return players[(idx + 1) % len(players)]

        # 3) Во всех остальных случаях ходит тот же игрок.
        return current_to_play

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        # Под Dominion с training_steps ~ 500k
        if trained_steps < 100e3:
            return 1.0   # сильная стохастика в самом начале
        elif trained_steps < 300e3:
            return 0.5   # более жадный поиск на среднем этапе
        else:
            return 0.25  # в конце почти всегда выбираем лучшую по визитам


# ----------------- AdapterGame -----------------
class AdapterGame(AbstractGame):
    """
    Phase-aware Dominion Base 2E adapter with STABLE action IDs.

    Key idea:
      - Action IDs are defined by card names, not by hand/supply indices.
      - This removes "action meaning drift" between episodes.
    """

    def __init__(
        self,
        num_players: int = 2,
        kingdom: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.num_players = num_players
        self.seed = seed
        self.rng = random.Random(seed)

        self.kingdom = kingdom  # if None -> random each reset

        self.dom = DominionGame(
            num_players=num_players,
            kingdom=self._sample_kingdom() if kingdom is None else kingdom,
            seed=seed,
            console_mode=False,
        )

        self.card_names: List[str] = sorted(list(self.dom.cards.keys()))
        self.n_card_types = len(self.card_names)

        # 4 блока зон игрока + supply + 5 скаляров (actions, buys, coins, phase_flag, turn_number)
        self.obs_len = self.n_card_types * 4 + self.n_card_types + 5
        self.observation_shape = (1, 1, self.obs_len)

        self.config = MuZeroConfig()
        self.config.observation_shape = self.observation_shape
        # два игрока для zero-sum MCTS
        self.config.players = list(range(self.num_players))

        self.last_actor: int = self.dom.to_play()
        # Глобальный номер хода (инкремент при успешном END_TURN)
        self.turn_number: int = 0

        # stable name->index mapping for actions
        self._all_cards: List[str] = list(ALL_CARDS)
        self._name_to_all_idx: Dict[str, int] = {n: i for i, n in enumerate(self._all_cards)}

        self._tb_stats = None
        self._episode_index = 0


    def _sample_kingdom(self) -> List[str]:
        return self.rng.sample(BASE_2E_KINGDOM_CARDS, 10)

    # ----------------- Helpers -----------------

    def _select_id(self, name: str) -> int:
        return ACTION_SELECT_OFFSET + self._name_to_all_idx[name]

    def _supply_id(self, name: str) -> int:
        return ACTION_SUPPLY_OFFSET + self._name_to_all_idx[name]

    @staticmethod
    def _find_index_by_name(cards, name: str, pred=None) -> Optional[int]:
        for i, c in enumerate(cards):
            if c.name != name:
                continue
            if pred is not None and not pred(c):
                continue
            return i
        return None

    # ----------------- AbstractGame API -----------------

    def reset(self):
        kingdom = self.kingdom if self.kingdom is not None else self._sample_kingdom()

        self.dom = DominionGame(
            num_players=self.num_players,
            kingdom=kingdom,
            seed=self.seed,
            console_mode=False,
        )

        self.card_names = sorted(list(self.dom.cards.keys()))
        self.n_card_types = len(self.card_names)

        self.obs_len = self.n_card_types * 4 + self.n_card_types + 5
        self.observation_shape = (1, 1, self.obs_len)
        self.config.observation_shape = self.observation_shape

        self.last_actor = self.dom.to_play()
        self.turn_number = 0
        self._episode_index += 1
        self._tb_stats = self._init_episode_stats()
        return self.get_observation()

    # ----------------- TensorBoard episode stats -----------------

    def _init_episode_stats(self) -> dict:
        # накопители за игру
        return {
            "episode": self._episode_index,

            # 4) полные ходы (заканчиваются ACTION_END_TURN, у тебя это действие end_turn_and_advance())
            "full_turns": 0,

            # 7) суммарно не потраченные деньги на фазе покупки (на конец BUY перед end_turn)
            "unspent_coins_sum": {0: 0.0, 1: 0.0},

            # 8) среднее число action-карт, сыгранных в action-фазе
            "actions_played_total": {0: 0, 1: 0},
            "action_phases_count": {0: 0, 1: 0},
            "actions_played_this_action_phase": {0: 0, 1: 0},

            # 9) причина окончания игры
            "ended_by_provinces": 0,
            "ended_by_3_piles": 0,
        }

    def _player_all_cards(self, p) -> list:
        # все карты игрока в сумме
        return list(p.deck) + list(p.hand) + list(p.discard) + list(p.in_play)

    def _count_card_name(self, p, name: str) -> int:
        return sum(1 for c in self._player_all_cards(p) if c.name == name)

    def _count_kingdom_cards(self, p) -> int:
        # только 10 карт королевства текущей партии
        kingdom_set = set(self.dom.kingdom)  # в dominion_base обычно хранится список имён
        return sum(1 for c in self._player_all_cards(p) if c.name in kingdom_set)

    def _treasure_value_sum(self, p) -> float:
        # суммарная "стоимость сокровищ" в колоде игрока (по cost; если у тебя есть другой атрибут — поменяй)
        total = 0.0
        for c in self._player_all_cards(p):
            if c.has_type(CardType.TREASURE):
                total += float(getattr(c, "cost", 0))
        return total

    def _snapshot_endgame_metrics(self) -> dict:
        # 1) VP у каждого игрока
        scores = self.dom.compute_scores()  # {player_idx: vp}
        out = {
            "vp": {0: float(scores.get(0, 0)), 1: float(scores.get(1, 0))},
        }

        # 2) кол-во провинций у каждого игрока
        out["provinces"] = {
            0: float(self._count_card_name(self.dom.players[0], "Province")),
            1: float(self._count_card_name(self.dom.players[1], "Province")),
        }

        # 3) кол-во карт у каждого игрока
        out["cards_total"] = {
            0: float(len(self._player_all_cards(self.dom.players[0]))),
            1: float(len(self._player_all_cards(self.dom.players[1]))),
        }

        # 5) (сумма значений сокровищ) / (общее число карт) у каждого игрока
        out["treasure_value_per_card"] = {}
        for i in [0, 1]:
            p = self.dom.players[i]
            denom = max(1, len(self._player_all_cards(p)))
            out["treasure_value_per_card"][i] = float(self._treasure_value_sum(p)) / float(denom)

        # 6) кол-во карт из 10 уникальных королевских
        out["kingdom_cards_total"] = {
            0: float(self._count_kingdom_cards(self.dom.players[0])),
            1: float(self._count_kingdom_cards(self.dom.players[1])),
        }

        # 7) накоплено в self._tb_stats["unspent_coins_sum"]
        out["unspent_coins_sum"] = {
            0: float(self._tb_stats["unspent_coins_sum"][0]),
            1: float(self._tb_stats["unspent_coins_sum"][1]),
        }

        # 8) среднее actions played на action-phase
        out["avg_actions_played_in_action_phase"] = {}
        for i in [0, 1]:
            n = self._tb_stats["action_phases_count"][i]
            out["avg_actions_played_in_action_phase"][i] = (
                float(self._tb_stats["actions_played_total"][i]) / float(max(1, n))
            )

        # 4) full turns
        out["full_turns"] = float(self._tb_stats["full_turns"])

        # 9) причина окончания
        out["ended_by_provinces"] = float(self._tb_stats["ended_by_provinces"])
        out["ended_by_3_piles"] = float(self._tb_stats["ended_by_3_piles"])
        return out

    def get_episode_tensorboard_scalars(self) -> dict:
        """
        Вызывать ТОЛЬКО когда done=True (после is_game_over()).
        Возвращает плоский dict {tag: value} для TensorBoard.
        """
        snap = self._snapshot_endgame_metrics()

        scalars = {}
        for i in [0, 1]:
            scalars[f"dominion/vp_p{i}"] = snap["vp"][i]
            scalars[f"dominion/provinces_p{i}"] = snap["provinces"][i]
            scalars[f"dominion/cards_total_p{i}"] = snap["cards_total"][i]
            scalars[f"dominion/treasure_value_per_card_p{i}"] = snap["treasure_value_per_card"][i]
            scalars[f"dominion/kingdom_cards_total_p{i}"] = snap["kingdom_cards_total"][i]
            scalars[f"dominion/unspent_coins_sum_p{i}"] = snap["unspent_coins_sum"][i]
            scalars[f"dominion/avg_actions_played_action_phase_p{i}"] = snap["avg_actions_played_in_action_phase"][i]

        scalars["dominion/full_turns"] = snap["full_turns"]
        scalars["dominion/ended_by_provinces"] = snap["ended_by_provinces"]
        scalars["dominion/ended_by_3_piles"] = snap["ended_by_3_piles"]
        return scalars


    def close(self):
        pass

    def render(self):
        cur = self.dom.to_play()
        p = self.dom.players[cur]
        phase = self.dom.phase

        print(f"\n=== View: {p.name} (decision player) ===")
        print(f"Turn player: {self.dom.players[self.dom.turn_player].name}")
        print(f"Phase: {phase.name}")
        print(f"Actions/Buys/Coins: {self.dom.actions}/{self.dom.buys}/{self.dom.coins}")

        print("\nHand:")
        if not p.hand:
            print("  (empty)")
        else:
            for i, c in enumerate(p.hand):
                print(f"  [{i}] {c.name}")

        if p.discard:
            print("\nDiscard:")
            for i, c in enumerate(p.discard):
                print(f"  [{i}] {c.name}")

        print("\nSupply (current game):")
        for name in sorted(self.dom.supply.keys()):
            left = self.dom.supply.get(name, 0)
            cost = self.dom.cards[name].cost
            print(f"  {name}: {left} (cost {cost})")

        legal = self.legal_actions()

        print("\nLegal actions (id -> meaning):")
        if not legal:
            print("  (no legal actions?)")
            return

        for a in legal:
            print(f"  [{a}] {self.action_to_string(a)}")

        # explicit labels for control ids
        print(f"\nControl ids: END_ACTION={ACTION_END_ACTION}, END_TURN={ACTION_END_TURN}")

    def to_play(self) -> int:
        return self.dom.to_play()

    # ----------------- Legal actions -----------------

    def legal_actions(self) -> List[int]:
        legal: Set[int] = set()
        cur = self.dom.to_play()
        p = self.dom.players[cur]
        phase = self.dom.phase

        def add_select_from_hand(filter_fn=None):
            names = set()
            for c in p.hand:
                if filter_fn is None or filter_fn(c):
                    if c.name in self._name_to_all_idx:
                        names.add(c.name)
            for n in names:
                legal.add(self._select_id(n))

        def add_select_from_discard():
            names = {c.name for c in p.discard if c.name in self._name_to_all_idx}
            for n in names:
                legal.add(self._select_id(n))

        def add_supply_names(card_pred=None, max_cost: Optional[int] = None):
            for name, left in self.dom.supply.items():
                if left <= 0:
                    continue
                card = self.dom.cards.get(name)
                if not card:
                    continue
                if max_cost is not None and card.cost > max_cost:
                    continue
                if card_pred is not None and not card_pred(card):
                    continue
                if name in self._name_to_all_idx:
                    legal.add(self._supply_id(name))

        # ---------------- Phase-based legality ----------------

        if phase == TurnPhase.ACTION:
            if self.dom.actions > 0:
                add_select_from_hand(lambda c: c.has_type(CardType.ACTION))
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.BUY:
            if self.dom.buys > 0:
                add_supply_names(lambda c: c.cost <= self.dom.coins)
            # optional manual treasures
            add_select_from_hand(lambda c: c.has_type(CardType.TREASURE))
            legal.add(ACTION_END_TURN)

        elif phase == TurnPhase.CELLAR_DISCARD:
            if p.hand:
                add_select_from_hand()
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.CHAPEL_TRASH:
            if p.hand:
                add_select_from_hand()
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.HARBINGER_TOPDECK:
            if p.discard:
                add_select_from_discard()
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.VASSAL_DECIDE:
            legal.add(CHOICE_0)  # play revealed action
            legal.add(ACTION_END_ACTION)  # discard revealed action

        elif phase == TurnPhase.WORKSHOP_GAIN:
            legal.add(ACTION_END_ACTION)
            add_supply_names(lambda c: c.cost <= 4)

        elif phase == TurnPhase.BUREAUCRAT_TOPDECK_VICTORY:
            add_select_from_hand(lambda c: c.has_type(CardType.VICTORY))

        elif phase == TurnPhase.MILITIA_DISCARD:
            if len(p.hand) > 3:
                add_select_from_hand()

        elif phase == TurnPhase.MONEYLENDER_TRASH_COPPER:
            add_select_from_hand(lambda c: c.name == "Copper")
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.POACHER_DISCARD:
            if p.hand:
                add_select_from_hand()

        elif phase == TurnPhase.REMODEL_TRASH:
            if p.hand:
                add_select_from_hand()
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.REMODEL_GAIN:
            max_cost = 0
            if self.dom.pending and "max_cost" in self.dom.pending.data:
                max_cost = int(self.dom.pending.data["max_cost"])
            add_supply_names(lambda c: c.cost <= max_cost)
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.THRONE_ROOM_CHOOSE:
            add_select_from_hand(lambda c: c.has_type(CardType.ACTION))
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.LIBRARY_DECIDE:
            legal.add(CHOICE_0)  # keep
            legal.add(ACTION_END_ACTION)  # set aside

        elif phase == TurnPhase.MINE_TRASH:
            add_select_from_hand(lambda c: c.has_type(CardType.TREASURE))
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.MINE_GAIN:
            max_cost = 0
            if self.dom.pending and "max_cost" in self.dom.pending.data:
                max_cost = int(self.dom.pending.data["max_cost"])
            add_supply_names(lambda c: c.has_type(CardType.TREASURE) and c.cost <= max_cost)
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.SENTRY_DECIDE_ONE:
            legal.update([CHOICE_0, CHOICE_1, CHOICE_2])  # trash/discard/keep

        elif phase == TurnPhase.SENTRY_ORDER:
            legal.update([CHOICE_0, CHOICE_1])  # pick index 0 or 1 in keep list

        elif phase == TurnPhase.ARTISAN_GAIN:
            add_supply_names(lambda c: c.cost <= 5)
            legal.add(ACTION_END_ACTION)

        elif phase == TurnPhase.ARTISAN_TOPDECK:
            if p.hand:
                add_select_from_hand()
            legal.add(ACTION_END_ACTION)

        return sorted(a for a in legal if 0 <= a < ACTION_SPACE_SIZE)

    # ----------------- Step -----------------

    def step(self, action: int):
        reward = 0.0
        done = False

        # текущий "decision player"
        cur = self.dom.to_play()
        self.last_actor = cur
        phase = self.dom.phase

        # --- track action-phase start / action plays ---
        if self._tb_stats is not None:
            tp = self.dom.turn_player  # чей сейчас ход
            # фиксируем начало action-фазы (1 раз на фазу)
            if phase == TurnPhase.ACTION and self.dom.actions == 1 and self.dom.buys >= 1 and self.dom.coins == 0:
                # это эвристика; если есть более точный сигнал "entered phase" — лучше использовать его
                self._tb_stats["action_phases_count"][tp] += 1
                self._tb_stats["actions_played_this_action_phase"][tp] = 0

        # --- Potential до действия (Phi_before) ---
        phi_before = self._state_potential(cur)

        ok = False

        # -------------- Select-by-name --------------
        if ACTION_SELECT_OFFSET <= action < ACTION_SELECT_OFFSET + N_CARDS:
            name = self._all_cards[action - ACTION_SELECT_OFFSET]
            ok = self._handle_select_by_name(cur, phase, name)

        # -------------- Supply-by-name --------------
        elif ACTION_SUPPLY_OFFSET <= action < ACTION_SUPPLY_OFFSET + N_CARDS:
            name = self._all_cards[action - ACTION_SUPPLY_OFFSET]
            ok = self._handle_supply_by_name(cur, phase, name)

        # -------------- Choices --------------
        elif action in (CHOICE_0, CHOICE_1, CHOICE_2):
            ok = self._handle_choice(cur, phase, action)

        # -------------- End/confirm/skip --------------
        elif action == ACTION_END_ACTION:
            ok = self._handle_end_action(cur, phase)

        # -------------- End turn --------------
        elif action == ACTION_END_TURN:
            tp_before = self.dom.turn_player
            coins_before = float(self.dom.coins)
            ok = (phase == TurnPhase.BUY) and self.dom.end_turn_and_advance()
            if ok and self._tb_stats is not None:
                self._tb_stats["full_turns"] += 1
                self._tb_stats["unspent_coins_sum"][tp_before] += coins_before

        # штраф за нелегальное действие
        if not ok:
            reward += REWARD_ILLEGAL_ACTION

        # --- Potential после действия (Phi_after) ---
        phi_after = self._state_potential(cur)

        # Potential-based shaping: alpha * (Phi_after - Phi_before)
        delta_phi = phi_after - phi_before
        reward += REWARD_SHAPING_ALPHA * delta_phi

        # лёгкий штраф за длину партии
        reward += REWARD_STEP_PENALTY

        # терминальная награда поверх шейпинга
        if self.dom.is_game_over():
            done = True
            reward += self._terminal_reward_for_last_actor()

            if self._tb_stats is not None:
                # конец по Province pile?
                prov_left = int(self.dom.supply.get("Province", 9999))
                if prov_left <= 0:
                    self._tb_stats["ended_by_provinces"] = 1
                # конец по 3 пустым поставкам
                empty_piles = sum(1 for _, left in self.dom.supply.items() if int(left) <= 0)
                if empty_piles >= 3:
                    self._tb_stats["ended_by_3_piles"] = 1

        obs = self.get_observation()
        return obs, float(reward), bool(done)

    # ----------------- Select dispatcher -----------------

    def _handle_select_by_name(self, player: int, phase: TurnPhase, name: str) -> bool:
        p = self.dom.players[player]

        if phase == TurnPhase.ACTION:
            idx = self._find_index_by_name(p.hand, name, lambda c: c.has_type(CardType.ACTION))
            if idx is None:
                return False
            ok = self.dom.play_action_from_hand(player, idx)
            if ok and self._tb_stats is not None:
                tp = self.dom.turn_player
                self._tb_stats["actions_played_total"][tp] += 1
                self._tb_stats["actions_played_this_action_phase"][tp] += 1
            return ok

        if phase == TurnPhase.BUY:
            idx = self._find_index_by_name(p.hand, name, lambda c: c.has_type(CardType.TREASURE))
            return False if idx is None else self.dom.play_treasure_from_hand(player, idx)

        if phase == TurnPhase.CELLAR_DISCARD:
            idx = self._find_index_by_name(p.hand, name)
            return False if idx is None else self.dom.cellar_discard_one(idx)

        if phase == TurnPhase.CHAPEL_TRASH:
            idx = self._find_index_by_name(p.hand, name)
            return False if idx is None else self.dom.chapel_trash_one(idx)

        if phase == TurnPhase.HARBINGER_TOPDECK:
            idx = self._find_index_by_name(p.discard, name)
            return False if idx is None else self.dom.harbinger_topdeck(idx)

        if phase == TurnPhase.BUREAUCRAT_TOPDECK_VICTORY:
            idx = self._find_index_by_name(p.hand, name, lambda c: c.has_type(CardType.VICTORY))
            return False if idx is None else self.dom.bureaucrat_topdeck_victory(idx)

        if phase == TurnPhase.MILITIA_DISCARD:
            idx = self._find_index_by_name(p.hand, name)
            return False if idx is None else self.dom.militia_discard_one(idx)

        if phase == TurnPhase.MONEYLENDER_TRASH_COPPER:
            idx = self._find_index_by_name(p.hand, name, lambda c: c.name == "Copper")
            return False if idx is None else self.dom.moneylender_trash_copper(idx)

        if phase == TurnPhase.POACHER_DISCARD:
            idx = self._find_index_by_name(p.hand, name)
            return False if idx is None else self.dom.poacher_discard_one(idx)

        if phase == TurnPhase.REMODEL_TRASH:
            idx = self._find_index_by_name(p.hand, name)
            return False if idx is None else self.dom.remodel_trash(idx)

        if phase == TurnPhase.MINE_TRASH:
            idx = self._find_index_by_name(p.hand, name, lambda c: c.has_type(CardType.TREASURE))
            return False if idx is None else self.dom.mine_trash(idx)

        if phase == TurnPhase.THRONE_ROOM_CHOOSE:
            idx = self._find_index_by_name(p.hand, name, lambda c: c.has_type(CardType.ACTION))
            return False if idx is None else self.dom.throne_room_choose(idx)

        if phase == TurnPhase.ARTISAN_TOPDECK:
            if not p.hand:
                return self.dom.artisan_topdeck(0)
            idx = self._find_index_by_name(p.hand, name)
            return False if idx is None else self.dom.artisan_topdeck(idx)

        return False

    # ----------------- Supply dispatcher -----------------

    def _handle_supply_by_name(self, player: int, phase: TurnPhase, name: str) -> bool:
        if phase == TurnPhase.BUY:
            return self.dom.buy_card(player, name)

        if phase == TurnPhase.WORKSHOP_GAIN:
            return self.dom.workshop_gain(name)

        if phase == TurnPhase.REMODEL_GAIN:
            return self.dom.remodel_gain(name)

        if phase == TurnPhase.MINE_GAIN:
            return self.dom.mine_gain(name)

        if phase == TurnPhase.ARTISAN_GAIN:
            return self.dom.artisan_gain(name)

        return False

    # ----------------- Choice dispatcher -----------------

    def _handle_choice(self, player: int, phase: TurnPhase, action: int) -> bool:
        if phase == TurnPhase.VASSAL_DECIDE:
            if action == CHOICE_0:
                return self.dom.vassal_play_revealed()
            return False

        if phase == TurnPhase.LIBRARY_DECIDE:
            if action == CHOICE_0:
                return self.dom.library_keep()
            return False

        if phase == TurnPhase.SENTRY_DECIDE_ONE:
            if action == CHOICE_0:
                return self.dom.sentry_choose_trash()
            if action == CHOICE_1:
                return self.dom.sentry_choose_discard()
            if action == CHOICE_2:
                return self.dom.sentry_choose_keep()
            return False

        if phase == TurnPhase.SENTRY_ORDER:
            if action == CHOICE_0:
                return self.dom.sentry_order_choose(0)
            if action == CHOICE_1:
                return self.dom.sentry_order_choose(1)
            return False

        return False

    # ----------------- End/confirm dispatcher -----------------

    def _handle_end_action(self, player: int, phase: TurnPhase) -> bool:
        if phase == TurnPhase.ACTION:
            ok = self.dom.end_action_phase()
            if ok:
                self._auto_play_treasures()
            return ok

        if phase == TurnPhase.CELLAR_DISCARD:
            self.dom.cellar_finish()
            return True

        if phase == TurnPhase.CHAPEL_TRASH:
            self.dom.chapel_finish()
            return True

        if phase == TurnPhase.HARBINGER_TOPDECK:
            self.dom.harbinger_skip()
            return True

        if phase == TurnPhase.VASSAL_DECIDE:
            return self.dom.vassal_discard_revealed()

        if phase == TurnPhase.WORKSHOP_GAIN:
            self.dom.workshop_skip()
            return True

        if phase == TurnPhase.MONEYLENDER_TRASH_COPPER:
            self.dom.moneylender_skip()
            return True

        if phase in (TurnPhase.REMODEL_TRASH, TurnPhase.REMODEL_GAIN):
            self.dom.remodel_skip()
            return True

        if phase == TurnPhase.THRONE_ROOM_CHOOSE:
            self.dom.throne_room_skip()
            return True

        if phase == TurnPhase.LIBRARY_DECIDE:
            return self.dom.library_set_aside()

        if phase in (TurnPhase.MINE_TRASH, TurnPhase.MINE_GAIN):
            self.dom.mine_skip()
            return True

        if phase == TurnPhase.ARTISAN_GAIN:
            self.dom.artisan_gain_skip()
            return True

        if phase == TurnPhase.ARTISAN_TOPDECK:
            # safe fallback
            return self.dom.artisan_topdeck(0)

        return False

    # ----------------- Auto treasures -----------------

    def _auto_play_treasures(self) -> None:
        if self.dom.phase != TurnPhase.BUY:
            return
        tp = self.dom.turn_player
        p = self.dom.players[tp]
        while True:
            t_idx = next((i for i, c in enumerate(p.hand) if c.has_type(CardType.TREASURE)), None)
            if t_idx is None:
                break
            self.dom.play_treasure_from_hand(tp, t_idx)

    # ----------------- Observation -----------------

    def get_observation(self):
        cur = self.dom.to_play()
        p = self.dom.players[cur]

        name_to_index = {n: i for i, n in enumerate(self.card_names)}

        deck_counts = np.zeros(self.n_card_types, dtype=np.float32)
        hand_counts = np.zeros(self.n_card_types, dtype=np.float32)
        discard_counts = np.zeros(self.n_card_types, dtype=np.float32)
        in_play_counts = np.zeros(self.n_card_types, dtype=np.float32)
        supply_counts = np.zeros(self.n_card_types, dtype=np.float32)

        for c in p.deck:
            idx = name_to_index.get(c.name)
            if idx is not None:
                deck_counts[idx] += 1.0
        for c in p.hand:
            idx = name_to_index.get(c.name)
            if idx is not None:
                hand_counts[idx] += 1.0
        for c in p.discard:
            idx = name_to_index.get(c.name)
            if idx is not None:
                discard_counts[idx] += 1.0
        for c in p.in_play:
            idx = name_to_index.get(c.name)
            if idx is not None:
                in_play_counts[idx] += 1.0

        for i, name in enumerate(self.card_names):
            supply_counts[i] = float(self.dom.supply.get(name, 0))

        phase_flag = 1.0 if self.dom.phase == TurnPhase.BUY else 0.0
        scalars = np.array(
            [
                float(self.dom.actions),
                float(self.dom.buys),
                float(self.dom.coins),
                phase_flag,
                float(self.turn_number / self.config.max_moves)
            ],
            dtype=np.float32,
        )

        vec = np.concatenate(
            [deck_counts, hand_counts, discard_counts, in_play_counts, supply_counts, scalars],
            axis=0,
        )
        obs_len = self.n_card_types * 4 + self.n_card_types + 5
        assert vec.size == obs_len
        return vec.reshape((1, 1, obs_len)).astype(np.float32)

    # ----------------- VP diff helpers & terminal reward -----------------

    def _vp_diff_for_player(self, player_index: int) -> float:
        """
        Разница VP текущего игрока и лучшего из остальных.
        Используется для step-wise reward shaping.
        """
        scores = self.dom.compute_scores()
        my_score = float(scores.get(player_index, 0))
        others = [float(v) for k, v in scores.items() if k != player_index]
        if not others:
            return 0.0
        best_other = max(others)
        return my_score - best_other
    
    def _state_potential(self, player_index: int) -> float:
        """
        Потенциал состояния для shaping:
        Phi = tanh( vp_diff / REWARD_SHAPING_NORM )
        Ограничивает вклад очень больших отрывов по VP.
        """
        vp_diff = self._vp_diff_for_player(player_index)
        x = vp_diff / REWARD_SHAPING_NORM
        return math.tanh(x)

    def _terminal_reward_for_last_actor(self) -> float:
        scores = self.dom.compute_scores()
        last = self.last_actor
        last_score = scores.get(last, 0)
        best_other = max(v for k, v in scores.items() if k != last)
        if last_score > best_other:
            return 1.0
        if last_score == best_other:
            return 0.0
        return -1.0

    # ----------------- Debug string -----------------

    def action_to_string(self, action_number: int) -> str:
        phase = self.dom.phase

        if ACTION_SELECT_OFFSET <= action_number < ACTION_SELECT_OFFSET + N_CARDS:
            name = self._all_cards[action_number - ACTION_SELECT_OFFSET]
            if phase == TurnPhase.ACTION:
                return f"SELECT {name} -> play Action from hand"
            if phase == TurnPhase.BUY:
                return f"SELECT {name} -> play Treasure from hand"
            if phase in (TurnPhase.CELLAR_DISCARD, TurnPhase.MILITIA_DISCARD, TurnPhase.POACHER_DISCARD):
                return f"SELECT {name} -> discard from hand"
            if phase in (TurnPhase.CHAPEL_TRASH, TurnPhase.REMODEL_TRASH, TurnPhase.MINE_TRASH, TurnPhase.MONEYLENDER_TRASH_COPPER):
                return f"SELECT {name} -> trash/choose from hand"
            if phase == TurnPhase.HARBINGER_TOPDECK:
                return f"SELECT {name} -> topdeck from discard"
            if phase == TurnPhase.BUREAUCRAT_TOPDECK_VICTORY:
                return f"SELECT {name} -> topdeck Victory from hand"
            if phase == TurnPhase.THRONE_ROOM_CHOOSE:
                return f"SELECT {name} -> Throne Room target"
            if phase == TurnPhase.ARTISAN_TOPDECK:
                return f"SELECT {name} -> topdeck from hand"
            return f"SELECT {name} (phase {phase.name})"

        if ACTION_SUPPLY_OFFSET <= action_number < ACTION_SUPPLY_OFFSET + N_CARDS:
            name = self._all_cards[action_number - ACTION_SUPPLY_OFFSET]
            if phase == TurnPhase.BUY:
                return f"SUPPLY {name} -> buy"
            if phase in (TurnPhase.WORKSHOP_GAIN, TurnPhase.REMODEL_GAIN, TurnPhase.MINE_GAIN, TurnPhase.ARTISAN_GAIN):
                return f"SUPPLY {name} -> gain"
            return f"SUPPLY {name} (phase {phase.name})"

        if action_number == CHOICE_0:
            if phase == TurnPhase.VASSAL_DECIDE:
                return "CHOICE_0 -> Vassal: PLAY revealed Action"
            if phase == TurnPhase.LIBRARY_DECIDE:
                return "CHOICE_0 -> Library: KEEP drawn Action"
            if phase == TurnPhase.SENTRY_DECIDE_ONE:
                return "CHOICE_0 -> Sentry: TRASH"
            if phase == TurnPhase.SENTRY_ORDER:
                return "CHOICE_0 -> Sentry order: pick keep[0]"
            return "CHOICE_0"

        if action_number == CHOICE_1:
            if phase == TurnPhase.SENTRY_DECIDE_ONE:
                return "CHOICE_1 -> Sentry: DISCARD"
            if phase == TurnPhase.SENTRY_ORDER:
                return "CHOICE_1 -> Sentry order: pick keep[1]"
            return "CHOICE_1"

        if action_number == CHOICE_2:
            if phase == TurnPhase.SENTRY_DECIDE_ONE:
                return "CHOICE_2 -> Sentry: KEEP"
            return "CHOICE_2"

        if action_number == ACTION_END_ACTION:
            if phase == TurnPhase.VASSAL_DECIDE:
                return "END_ACTION -> Vassal: DISCARD revealed Action"
            if phase == TurnPhase.LIBRARY_DECIDE:
                return "END_ACTION -> Library: SET ASIDE drawn Action"
            return f"END_ACTION -> End/Confirm/Skip ({phase.name})"

        if action_number == ACTION_END_TURN:
            return "END_TURN"

        return f"Unknown {action_number}"


Game = AdapterGame
