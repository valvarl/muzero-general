from __future__ import annotations

import datetime
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
REWARD_VP_DIFF_SCALE = 0.25      # шаговый шейпинг: 0.25 * (VP_diff_after - VP_diff_before)
REWARD_ILLEGAL_ACTION = -0.1     # небольшой штраф за нелегальное действие
REWARD_STEP_PENALTY = -0.002     # штраф за "затягивание" партии на каждый шаг

# ----------------- MuZero-like config -----------------
class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None

        self.observation_shape = (1, 1, 169)  # set by adapter
        self.action_space = list(range(ACTION_SPACE_SIZE))

        # true 2-player env
        self.players = [0]
        self.stacked_observations = 0

        self.muzero_player = 0
        self.opponent = "self"

        self.num_workers = 4
        self.selfplay_on_gpu = False
        self.max_moves = 600
        self.num_simulations = 50
        self.discount = 1
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.network = "resnet"
        self.support_size = 10

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
        self.training_steps = 150000
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = True

        self.optimizer = "SGD"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.03
        self.lr_decay_rate = 0.75
        self.lr_decay_steps = 150000

        self.replay_buffer_size = 10000
        self.num_unroll_steps = 50
        self.td_steps = 200
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None

    def next_to_play(self, current_to_play: int, action: int) -> int:
        return current_to_play

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


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

        self.obs_len = self.n_card_types * 4 + self.n_card_types + 4
        self.observation_shape = (1, 1, self.obs_len)

        self.config = MuZeroConfig()
        self.config.observation_shape = self.observation_shape

        self.last_actor: int = self.dom.to_play()

        # stable name->index mapping for actions
        self._all_cards: List[str] = list(ALL_CARDS)
        self._name_to_all_idx: Dict[str, int] = {n: i for i, n in enumerate(self._all_cards)}

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

        self.obs_len = self.n_card_types * 4 + self.n_card_types + 4
        self.observation_shape = (1, 1, self.obs_len)
        self.config.observation_shape = self.observation_shape

        self.last_actor = self.dom.to_play()
        return self.get_observation()

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

        # VP-разница ДО действия (с точки зрения текущего игрока)
        vp_diff_before = self._vp_diff_for_player(cur)

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
            ok = (phase == TurnPhase.BUY) and self.dom.end_turn_and_advance()

        # штраф за нелегальное действие
        if not ok:
            reward += REWARD_ILLEGAL_ACTION

        # VP-разница ПОСЛЕ действия (всё ещё с точки зрения cur)
        vp_diff_after = self._vp_diff_for_player(cur)
        delta_diff = vp_diff_after - vp_diff_before
        reward += REWARD_VP_DIFF_SCALE * delta_diff

        # лёгкий штраф за длину партии
        reward += REWARD_STEP_PENALTY

        # терминальная награда поверх шейпинга
        if self.dom.is_game_over():
            done = True
            reward += self._terminal_reward_for_last_actor()

        obs = self.get_observation()
        return obs, float(reward), bool(done)

    # ----------------- Select dispatcher -----------------

    def _handle_select_by_name(self, player: int, phase: TurnPhase, name: str) -> bool:
        p = self.dom.players[player]

        if phase == TurnPhase.ACTION:
            idx = self._find_index_by_name(p.hand, name, lambda c: c.has_type(CardType.ACTION))
            return False if idx is None else self.dom.play_action_from_hand(player, idx)

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
            [float(self.dom.actions), float(self.dom.buys), float(self.dom.coins), phase_flag],
            dtype=np.float32,
        )

        vec = np.concatenate(
            [deck_counts, hand_counts, discard_counts, in_play_counts, supply_counts, scalars],
            axis=0,
        )
        obs_len = self.n_card_types * 4 + self.n_card_types + 4
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
            # на случай каких-то экзотических конфигураций, но в 2p Dominion это не должно случиться
            return 0.0
        best_other = max(others)
        return my_score - best_other

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
