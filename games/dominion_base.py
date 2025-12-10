"""
(2) [action] Cellar: Discard any number of cards. +1 Card per card discarded.
(2) [action] Chapel: Trash up to 4 cards from your hand.
(2) [action-reaction] Moat: +2 Cards. When another player plays an Attack card, you may first reveal this from your hand, to be unaffected by it.
(3) [action] Harbinger: +1 Card. +1 Action. Look through your discard pile. You may put a card from it onto your deck.
(3) [action] Merchant: +1 Card. +1 Action. The first time you play a Silver this turn, +$1.
(3) [action] Vassal: +$2. Discard the top card of your deck. If it's an Action card, you may play it.
(3) [action] Village: +1 Card. +2 Actions.
(3) [action] Workshop: Gain a card costing up to $4.
(4) [action-attack] Bureaucrat: Gain a Silver onto your deck. Each other player reveals a Victory card from their hand and puts it onto their deck (or reveals a hand with no Victory cards).
(4) [victory] Gardens: Worth $1 per 10 cards you have (round down).
(4) [action-attack] Militia: +$2. Each other player discards down to 3 cards in hand.
(4) [action] Moneylender: You may trash a Copper from your hand for +$3.
(4) [action] Poacher: +1 Card. +1 Action. +$1. Discard a card per empty Supply pile.
(4) [action] Remodel: Trash a card from your hand. Gain a card costing up to $2 more than it.
(4) [action] Smithy: +3 Cards.
(4) [action] Throne Room: You may play an Action card from your hand twice.
(5) [action-attack] Bandit: Gain a Gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper, and discards the rest.
(5) [action] Council Room: +4 Cards. +1 Buy. Each other player draws a card.
(5) [action] Festival: +2 Actions. +1 Buy. +$2.
(5) [action] Laboratory: +2 Cards. +1 Action.
(5) [action] Library: Draw until you have 7 cards in hand, skipping any Action cards you choose to; set those aside, discarding them afterwards.
(5) [action] Market: +1 Card. +1 Action. +1 Buy. +$1.
(5) [action] Mine: You may trash a Treasure from your hand. Gain a Treasure to your hand costing up to $3 more than it.
(5) [action] Sentry: +1 Card. +1 Action. Look at the top 2 cards of your deck. Trash and/or discard any number of them. Put the rest back on top in any order.
(5) [action-attack] Witch: +2 Cards. Each other player gains a Curse.
(6) [action] Artisan: Gain a card to your hand costing up to $5. Put a card from your hand onto your deck.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple


# -------------------- Card model --------------------


class CardType(Enum):
    ACTION = "Action"
    TREASURE = "Treasure"
    VICTORY = "Victory"
    CURSE = "Curse"
    ATTACK = "Attack"
    REACTION = "Reaction"


@dataclass(frozen=True)
class CardDef:
    name: str
    cost: int
    types: Tuple[CardType, ...]
    plus_cards: int = 0
    plus_actions: int = 0
    plus_buys: int = 0
    plus_coins: int = 0
    on_play: Optional[Callable["Game", int, "CardDef"]] = None

    def has_type(self, t: CardType) -> bool:
        return t in self.types


@dataclass
class PlayerState:
    name: str
    deck: List[CardDef]
    hand: List[CardDef]
    discard: List[CardDef]
    in_play: List[CardDef]


# -------------------- Phases --------------------


class TurnPhase(Enum):
    ACTION = auto()
    BUY = auto()

    # Generic / card-specific decision phases
    CELLAR_DISCARD = auto()
    CHAPEL_TRASH = auto()
    HARBINGER_TOPDECK = auto()
    VASSAL_DECIDE = auto()
    WORKSHOP_GAIN = auto()
    BUREAUCRAT_TOPDECK_VICTORY = auto()
    MILITIA_DISCARD = auto()
    MONEYLENDER_TRASH_COPPER = auto()
    POACHER_DISCARD = auto()
    REMODEL_TRASH = auto()
    REMODEL_GAIN = auto()
    THRONE_ROOM_CHOOSE = auto()
    LIBRARY_DECIDE = auto()
    MINE_TRASH = auto()
    MINE_GAIN = auto()

    SENTRY_DECIDE_ONE = auto()
    SENTRY_ORDER = auto()

    ARTISAN_GAIN = auto()
    ARTISAN_TOPDECK = auto()


@dataclass
class Pending:
    kind: TurnPhase
    player: int
    data: Dict[str, object] = field(default_factory=dict)


# -------------------- Base 2E kingdom list --------------------


BASE_2E_KINGDOM_CARDS: List[str] = [
    "Cellar",
    "Chapel",
    "Moat",
    "Harbinger",
    "Merchant",
    "Vassal",
    "Village",
    "Workshop",
    "Bureaucrat",
    "Gardens",
    "Militia",
    "Moneylender",
    "Poacher",
    "Remodel",
    "Smithy",
    "Throne Room",
    "Bandit",
    "Council Room",
    "Festival",
    "Laboratory",
    "Library",
    "Market",
    "Mine",
    "Sentry",
    "Witch",
    "Artisan",
]


# -------------------- Game --------------------


class Game:
    """
    RL-friendly Dominion Base 2E core.

    - Supports all 26 Base 2E kingdom cards.
    - Uses phase+pending state machine for ALL interactive effects.
    - Does not require console input for RL mode.
    - Attack reactions (Moat) are treated as automatic reveal/negation for simplicity.
    """

    def __init__(
        self,
        num_players: int = 2,
        kingdom: Optional[List[str]] = None,
        seed: Optional[int] = None,
        console_mode: bool = False,
    ):
        if num_players < 2:
            raise ValueError("Dominion needs at least 2 players")

        self.num_players = num_players
        self.rng = random.Random(seed)
        self.console_mode = console_mode

        self.cards: Dict[str, CardDef] = create_card_defs()
        self.players: List[PlayerState] = []
        self.supply: Dict[str, int] = {}
        self.trash: List[CardDef] = []

        # Turn economy (for current turn player)
        self.turn_player: int = 0
        self.actions: int = 1
        self.buys: int = 1
        self.coins: int = 0
        self.merchant_bonus_remaining: int = 0

        # Phase machine
        self.phase: TurnPhase = TurnPhase.ACTION
        self.pending: Optional[Pending] = None

        # Queues / contexts for multi-victim or multi-step cards
        self._militia_queue: List[int] = []
        self._bureaucrat_queue: List[int] = []
        self._poacher_remaining: int = 0

        self._library_ctx: Optional[Dict[str, object]] = None
        self._sentry_ctx: Optional[Dict[str, object]] = None

        # Deferred effect queue (used for Throne Room second play)
        self._deferred: List[Tuple[str, int, CardDef]] = []

        # Choose kingdom
        if kingdom is None:
            kingdom = self.rng.sample(BASE_2E_KINGDOM_CARDS, 10)
        self.kingdom = list(kingdom)

        self._setup_supply(self.kingdom)
        self._setup_players()

        self.start_turn(0)

    # -------------------- Setup --------------------

    def _setup_supply(self, kingdom: List[str]) -> None:
        n = self.num_players
        self.supply.clear()

        # basic treasures
        self.supply["Copper"] = 60 - 7 * n
        self.supply["Silver"] = 40
        self.supply["Gold"] = 30

        # basic victory
        victory_count = 8 if n == 2 else 12
        for name in ["Estate", "Duchy", "Province"]:
            self.supply[name] = victory_count

        # curses
        self.supply["Curse"] = 10 * (n - 1)

        # kingdom piles
        for name in kingdom:
            c = self.cards[name]
            if c.has_type(CardType.VICTORY):
                self.supply[name] = victory_count
            else:
                self.supply[name] = 10

    def _setup_players(self) -> None:
        self.players.clear()
        for i in range(self.num_players):
            deck = [self.cards["Copper"]] * 7 + [self.cards["Estate"]] * 3
            self.rng.shuffle(deck)
            player = PlayerState(
                name=f"P{i+1}",
                deck=deck,
                hand=[],
                discard=[],
                in_play=[],
            )
            self.players.append(player)
            self._draw_cards(i, 5)

    # -------------------- Zones --------------------

    def _draw_one(self, player_idx: int) -> Optional[CardDef]:
        p = self.players[player_idx]
        if not p.deck:
            if not p.discard:
                return None
            self.rng.shuffle(p.discard)
            p.deck = p.discard
            p.discard = []
        return p.deck.pop()

    def _draw_cards(self, player_idx: int, n: int) -> None:
        for _ in range(n):
            c = self._draw_one(player_idx)
            if c is None:
                break
            self.players[player_idx].hand.append(c)

    def gain(self, player_idx: int, card_name: str, where: str = "discard") -> bool:
        if self.supply.get(card_name, 0) <= 0:
            return False
        self.supply[card_name] -= 1
        card = self.cards[card_name]
        p = self.players[player_idx]
        if where == "discard":
            p.discard.append(card)
        elif where == "hand":
            p.hand.append(card)
        elif where == "topdeck":
            p.deck.append(card)
        else:
            raise ValueError(f"Unknown gain location: {where}")
        return True

    def trash_card_from_hand(self, player_idx: int, hand_index: int) -> Optional[CardDef]:
        p = self.players[player_idx]
        if not (0 <= hand_index < len(p.hand)):
            return None
        card = p.hand.pop(hand_index)
        self.trash.append(card)
        return card

    def discard_from_hand(self, player_idx: int, hand_indices: List[int]) -> None:
        p = self.players[player_idx]
        for idx in sorted(hand_indices, reverse=True):
            if 0 <= idx < len(p.hand):
                p.discard.append(p.hand.pop(idx))

    def other_players(self, player_idx: int) -> List[int]:
        return [
            (player_idx + i) % self.num_players
            for i in range(1, self.num_players)
        ]

    # -------------------- End conditions --------------------

    def empty_piles(self) -> int:
        return sum(1 for v in self.supply.values() if v == 0)

    def is_game_over(self) -> bool:
        if self.supply.get("Province", 0) == 0:
            return True
        return self.empty_piles() >= 3

    # -------------------- Turn control --------------------

    def start_turn(self, player_idx: int) -> None:
        self.turn_player = player_idx
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.merchant_bonus_remaining = 0

        self.phase = TurnPhase.ACTION
        self.pending = None

        self._militia_queue = []
        self._bureaucrat_queue = []
        self._poacher_remaining = 0
        self._library_ctx = None
        self._sentry_ctx = None
        self._deferred.clear()

    def end_turn(self, player_idx: int) -> None:
        p = self.players[player_idx]
        p.discard.extend(p.in_play)
        p.in_play = []
        p.discard.extend(p.hand)
        p.hand = []
        self._draw_cards(player_idx, 5)

    def to_play(self) -> int:
        if self.pending is not None:
            return self.pending.player
        return self.turn_player

    def end_action_phase(self) -> bool:
        if self.pending is not None:
            return False
        if self.phase != TurnPhase.ACTION:
            return False
        self.phase = TurnPhase.BUY
        return True

    def end_turn_and_advance(self) -> bool:
        if self.pending is not None:
            return False
        if self.phase != TurnPhase.BUY:
            return False
        cur = self.turn_player
        self.end_turn(cur)
        if self.is_game_over():
            return True
        nxt = (cur + 1) % self.num_players
        self.start_turn(nxt)
        return True

    # -------------------- Core play --------------------

    def _apply_card_effect(self, player_idx: int, card: CardDef) -> None:
        if card.plus_cards:
            self._draw_cards(player_idx, card.plus_cards)
        if card.plus_actions:
            self.actions += card.plus_actions
        if card.plus_buys:
            self.buys += card.plus_buys
        if card.plus_coins:
            self.coins += card.plus_coins
        if card.on_play:
            card.on_play(self, player_idx, card)

    def _clear_pending(self) -> None:
        self.pending = None
        # after a pending resolves, run deferred effects if any
        self._run_deferred_if_idle()

    def _run_deferred_if_idle(self) -> None:
        # Only run when no pending
        if self.pending is not None:
            return
        if not self._deferred:
            return
        op, player, card = self._deferred.pop(0)
        if op == "apply_effect":
            self._apply_card_effect(player, card)
            self._run_deferred_if_idle()

    # -------------------- Action / Treasure / Buy --------------------

    def play_action_from_hand(self, player_idx: int, hand_index: int, free: bool = False) -> bool:
        if self.pending is not None:
            return False
        if self.phase != TurnPhase.ACTION:
            return False
        if player_idx != self.turn_player:
            return False

        p = self.players[player_idx]
        if not (0 <= hand_index < len(p.hand)):
            return False
        card = p.hand[hand_index]
        if not card.has_type(CardType.ACTION):
            return False

        if not free:
            if self.actions <= 0:
                return False
            self.actions -= 1

        card = p.hand.pop(hand_index)
        p.in_play.append(card)
        self._apply_card_effect(player_idx, card)
        return True

    def play_treasure_from_hand(self, player_idx: int, hand_index: int) -> bool:
        if self.pending is not None:
            return False
        if self.phase != TurnPhase.BUY:
            return False
        if player_idx != self.turn_player:
            return False

        p = self.players[player_idx]
        if not (0 <= hand_index < len(p.hand)):
            return False
        card = p.hand[hand_index]
        if not card.has_type(CardType.TREASURE):
            return False

        card = p.hand.pop(hand_index)
        p.in_play.append(card)
        self._apply_card_effect(player_idx, card)

        if card.name == "Silver" and self.merchant_bonus_remaining > 0:
            self.coins += 1
            self.merchant_bonus_remaining -= 1
        return True

    def buy_card(self, player_idx: int, card_name: str) -> bool:
        if self.pending is not None:
            return False
        if self.phase != TurnPhase.BUY:
            return False
        if player_idx != self.turn_player:
            return False
        if self.buys <= 0:
            return False

        card = self.cards.get(card_name)
        if card is None:
            return False
        if self.supply.get(card_name, 0) <= 0:
            return False
        if card.cost > self.coins:
            return False

        self.buys -= 1
        self.coins -= card.cost
        self.gain(player_idx, card_name)
        return True

    # -------------------- Reactions / attacks --------------------

    def player_has_moat(self, player_idx: int) -> bool:
        return any(c.name == "Moat" for c in self.players[player_idx].hand)

    # -------------------- Interactive card phases --------------------

    def start_cellar(self, player: int) -> None:
        self.phase = TurnPhase.CELLAR_DISCARD
        self.pending = Pending(TurnPhase.CELLAR_DISCARD, player, {"discarded": 0})

    def cellar_discard_one(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.CELLAR_DISCARD:
            return False
        player = self.pending.player
        p = self.players[player]
        if not (0 <= hand_index < len(p.hand)):
            return False
        self.discard_from_hand(player, [hand_index])
        self.pending.data["discarded"] = int(self.pending.data["discarded"]) + 1
        return True

    def cellar_finish(self) -> None:
        if not self.pending or self.pending.kind != TurnPhase.CELLAR_DISCARD:
            return
        player = self.pending.player
        n = int(self.pending.data.get("discarded", 0))
        if n > 0:
            self._draw_cards(player, n)
        self.phase = TurnPhase.ACTION
        self._clear_pending()

    def start_chapel(self, player: int) -> None:
        self.phase = TurnPhase.CHAPEL_TRASH
        self.pending = Pending(TurnPhase.CHAPEL_TRASH, player, {"remaining": 4})

    def chapel_trash_one(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.CHAPEL_TRASH:
            return False
        remaining = int(self.pending.data.get("remaining", 0))
        if remaining <= 0:
            return False
        player = self.pending.player
        card = self.trash_card_from_hand(player, hand_index)
        if card is None:
            return False
        self.pending.data["remaining"] = remaining - 1
        return True

    def chapel_finish(self) -> None:
        if self.pending and self.pending.kind == TurnPhase.CHAPEL_TRASH:
            self.phase = TurnPhase.ACTION
            self._clear_pending()

    def start_harbinger(self, player: int) -> None:
        p = self.players[player]
        if not p.discard:
            return
        self.phase = TurnPhase.HARBINGER_TOPDECK
        self.pending = Pending(TurnPhase.HARBINGER_TOPDECK, player, {})

    def harbinger_topdeck(self, discard_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.HARBINGER_TOPDECK:
            return False
        player = self.pending.player
        p = self.players[player]
        if not (0 <= discard_index < len(p.discard)):
            return False
        card = p.discard.pop(discard_index)
        p.deck.append(card)
        self.phase = TurnPhase.ACTION
        self._clear_pending()
        return True

    def harbinger_skip(self) -> None:
        if self.pending and self.pending.kind == TurnPhase.HARBINGER_TOPDECK:
            self.phase = TurnPhase.ACTION
            self._clear_pending()

    def merchant_bonus(self) -> None:
        self.merchant_bonus_remaining += 1

    def start_vassal(self, player: int) -> None:
        top = self._draw_one(player)
        if top is None:
            return
        if top.has_type(CardType.ACTION):
            self.phase = TurnPhase.VASSAL_DECIDE
            self.pending = Pending(TurnPhase.VASSAL_DECIDE, player, {"card": top})
        else:
            self.players[player].discard.append(top)

    def vassal_play_revealed(self) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.VASSAL_DECIDE:
            return False
        player = self.pending.player
        card = self.pending.data.get("card")
        if not isinstance(card, CardDef):
            return False
        self.players[player].in_play.append(card)
        self.phase = TurnPhase.ACTION
        self._clear_pending()
        self._apply_card_effect(player, card)
        return True

    def vassal_discard_revealed(self) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.VASSAL_DECIDE:
            return False
        player = self.pending.player
        card = self.pending.data.get("card")
        if not isinstance(card, CardDef):
            return False
        self.players[player].discard.append(card)
        self.phase = TurnPhase.ACTION
        self._clear_pending()
        return True

    def start_workshop(self, player: int) -> None:
        self.phase = TurnPhase.WORKSHOP_GAIN
        self.pending = Pending(TurnPhase.WORKSHOP_GAIN, player, {"max_cost": 4})

    def workshop_gain(self, card_name: str) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.WORKSHOP_GAIN:
            return False
        player = self.pending.player
        max_cost = int(self.pending.data.get("max_cost", 0))
        card = self.cards.get(card_name)
        if not card or card.cost > max_cost:
            return False
        ok = self.gain(player, card_name)
        if ok:
            self.phase = TurnPhase.ACTION
            self._clear_pending()
        return ok

    def workshop_skip(self) -> None:
        if self.pending and self.pending.kind == TurnPhase.WORKSHOP_GAIN:
            self.phase = TurnPhase.ACTION
            self._clear_pending()

    def start_bureaucrat(self, attacker: int) -> None:
        self.gain(attacker, "Silver", where="topdeck")

        self._bureaucrat_queue = []
        for v in self.other_players(attacker):
            if self.player_has_moat(v):
                continue
            if any(c.has_type(CardType.VICTORY) for c in self.players[v].hand):
                self._bureaucrat_queue.append(v)

        if self._bureaucrat_queue:
            victim = self._bureaucrat_queue.pop(0)
            self.phase = TurnPhase.BUREAUCRAT_TOPDECK_VICTORY
            self.pending = Pending(TurnPhase.BUREAUCRAT_TOPDECK_VICTORY, victim, {})

    def bureaucrat_topdeck_victory(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.BUREAUCRAT_TOPDECK_VICTORY:
            return False
        victim = self.pending.player
        p = self.players[victim]
        if not (0 <= hand_index < len(p.hand)):
            return False
        card = p.hand[hand_index]
        if not card.has_type(CardType.VICTORY):
            return False
        card = p.hand.pop(hand_index)
        p.deck.append(card)

        if self._bureaucrat_queue:
            nxt = self._bureaucrat_queue.pop(0)
            self.phase = TurnPhase.BUREAUCRAT_TOPDECK_VICTORY
            self.pending = Pending(TurnPhase.BUREAUCRAT_TOPDECK_VICTORY, nxt, {})
        else:
            self.phase = TurnPhase.ACTION
            self._clear_pending()
        return True

    def start_militia(self, attacker: int) -> None:
        self._militia_queue = []
        for v in self.other_players(attacker):
            if self.player_has_moat(v):
                continue
            if len(self.players[v].hand) > 3:
                self._militia_queue.append(v)

        if self._militia_queue:
            victim = self._militia_queue.pop(0)
            self.phase = TurnPhase.MILITIA_DISCARD
            self.pending = Pending(TurnPhase.MILITIA_DISCARD, victim, {})

    def militia_discard_one(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.MILITIA_DISCARD:
            return False
        victim = self.pending.player
        p = self.players[victim]
        if len(p.hand) <= 3:
            return False
        if not (0 <= hand_index < len(p.hand)):
            return False
        self.discard_from_hand(victim, [hand_index])

        if len(p.hand) <= 3:
            if self._militia_queue:
                nxt = self._militia_queue.pop(0)
                self.phase = TurnPhase.MILITIA_DISCARD
                self.pending = Pending(TurnPhase.MILITIA_DISCARD, nxt, {})
            else:
                self.phase = TurnPhase.ACTION
                self._clear_pending()
        return True

    def start_moneylender(self, player: int) -> None:
        p = self.players[player]
        if not any(c.name == "Copper" for c in p.hand):
            return
        self.phase = TurnPhase.MONEYLENDER_TRASH_COPPER
        self.pending = Pending(TurnPhase.MONEYLENDER_TRASH_COPPER, player, {})

    def moneylender_trash_copper(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.MONEYLENDER_TRASH_COPPER:
            return False
        player = self.pending.player
        p = self.players[player]
        if not (0 <= hand_index < len(p.hand)):
            return False
        if p.hand[hand_index].name != "Copper":
            return False
        self.trash_card_from_hand(player, hand_index)
        self.coins += 3
        self.phase = TurnPhase.ACTION
        self._clear_pending()
        return True

    def moneylender_skip(self) -> None:
        if self.pending and self.pending.kind == TurnPhase.MONEYLENDER_TRASH_COPPER:
            self.phase = TurnPhase.ACTION
            self._clear_pending()

    def start_poacher(self, player: int) -> None:
        empties = self.empty_piles()
        if empties <= 0:
            return
        if not self.players[player].hand:
            return
        self._poacher_remaining = empties
        self.phase = TurnPhase.POACHER_DISCARD
        self.pending = Pending(TurnPhase.POACHER_DISCARD, player, {"remaining": empties})

    def poacher_discard_one(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.POACHER_DISCARD:
            return False
        player = self.pending.player
        p = self.players[player]
        remaining = int(self.pending.data.get("remaining", 0))
        if remaining <= 0:
            return False
        if not (0 <= hand_index < len(p.hand)):
            return False
        self.discard_from_hand(player, [hand_index])
        remaining -= 1
        self.pending.data["remaining"] = remaining
        if remaining <= 0 or not p.hand:
            self.phase = TurnPhase.ACTION
            self._clear_pending()
        return True

    def start_remodel(self, player: int) -> None:
        if not self.players[player].hand:
            return
        self.phase = TurnPhase.REMODEL_TRASH
        self.pending = Pending(TurnPhase.REMODEL_TRASH, player, {})

    def remodel_trash(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.REMODEL_TRASH:
            return False
        player = self.pending.player
        p = self.players[player]
        if not (0 <= hand_index < len(p.hand)):
            return False
        trashed = p.hand[hand_index]
        self.trash_card_from_hand(player, hand_index)
        max_cost = trashed.cost + 2
        self.phase = TurnPhase.REMODEL_GAIN
        self.pending = Pending(TurnPhase.REMODEL_GAIN, player, {"max_cost": max_cost})
        return True

    def remodel_gain(self, card_name: str) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.REMODEL_GAIN:
            return False
        player = self.pending.player
        max_cost = int(self.pending.data.get("max_cost", 0))
        card = self.cards.get(card_name)
        if not card or card.cost > max_cost:
            return False
        ok = self.gain(player, card_name)
        if ok:
            self.phase = TurnPhase.ACTION
            self._clear_pending()
        return ok

    def remodel_skip(self) -> None:
        if self.pending and self.pending.kind in (TurnPhase.REMODEL_TRASH, TurnPhase.REMODEL_GAIN):
            self.phase = TurnPhase.ACTION
            self._clear_pending()

    def start_throne_room(self, player: int) -> None:
        if not any(c.has_type(CardType.ACTION) for c in self.players[player].hand):
            return
        self.phase = TurnPhase.THRONE_ROOM_CHOOSE
        self.pending = Pending(TurnPhase.THRONE_ROOM_CHOOSE, player, {})

    def throne_room_choose(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.THRONE_ROOM_CHOOSE:
            return False
        player = self.pending.player
        p = self.players[player]
        if not (0 <= hand_index < len(p.hand)):
            return False
        card = p.hand[hand_index]
        if not card.has_type(CardType.ACTION):
            return False

        card = p.hand.pop(hand_index)
        p.in_play.append(card)

        self.phase = TurnPhase.ACTION
        self._clear_pending()
        self._apply_card_effect(player, card)
        self._deferred.append(("apply_effect", player, card))
        self._run_deferred_if_idle()
        return True

    def throne_room_skip(self) -> None:
        if self.pending and self.pending.kind == TurnPhase.THRONE_ROOM_CHOOSE:
            self.phase = TurnPhase.ACTION
            self._clear_pending()

    def start_mine(self, player: int) -> None:
        if not any(c.has_type(CardType.TREASURE) for c in self.players[player].hand):
            return
        self.phase = TurnPhase.MINE_TRASH
        self.pending = Pending(TurnPhase.MINE_TRASH, player, {})

    def mine_trash(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.MINE_TRASH:
            return False
        player = self.pending.player
        p = self.players[player]
        if not (0 <= hand_index < len(p.hand)):
            return False
        card = p.hand[hand_index]
        if not card.has_type(CardType.TREASURE):
            return False
        trashed = card
        self.trash_card_from_hand(player, hand_index)
        max_cost = trashed.cost + 3
        self.phase = TurnPhase.MINE_GAIN
        self.pending = Pending(TurnPhase.MINE_GAIN, player, {"max_cost": max_cost})
        return True

    def mine_gain(self, card_name: str) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.MINE_GAIN:
            return False
        player = self.pending.player
        max_cost = int(self.pending.data.get("max_cost", 0))
        card = self.cards.get(card_name)
        if not card or not card.has_type(CardType.TREASURE) or card.cost > max_cost:
            return False
        ok = self.gain(player, card_name, where="hand")
        if ok:
            self.phase = TurnPhase.ACTION
            self._clear_pending()
        return ok

    def mine_skip(self) -> None:
        if self.pending and self.pending.kind in (TurnPhase.MINE_TRASH, TurnPhase.MINE_GAIN):
            self.phase = TurnPhase.ACTION
            self._clear_pending()

    def start_library(self, player: int) -> None:
        self._library_ctx = {"player": player, "set_aside": []}
        self._library_continue()

    def _library_continue(self) -> None:
        if self._library_ctx is None:
            return
        player = int(self._library_ctx["player"])
        p = self.players[player]

        if self.pending is not None:
            return

        while len(p.hand) < 7:
            drawn = self._draw_one(player)
            if drawn is None:
                break
            if drawn.has_type(CardType.ACTION):
                self.phase = TurnPhase.LIBRARY_DECIDE
                self.pending = Pending(TurnPhase.LIBRARY_DECIDE, player, {"card": drawn})
                return
            else:
                p.hand.append(drawn)

        set_aside: List[CardDef] = list(self._library_ctx.get("set_aside", []))
        p.discard.extend(set_aside)
        self._library_ctx = None
        self.phase = TurnPhase.ACTION

    def library_keep(self) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.LIBRARY_DECIDE:
            return False
        player = self.pending.player
        card = self.pending.data.get("card")
        if not isinstance(card, CardDef):
            return False
        self.players[player].hand.append(card)
        self.phase = TurnPhase.ACTION
        self._clear_pending()
        self._library_continue()
        return True

    def library_set_aside(self) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.LIBRARY_DECIDE:
            return False
        player = self.pending.player
        card = self.pending.data.get("card")
        if not isinstance(card, CardDef):
            return False
        if self._library_ctx is not None:
            self._library_ctx["set_aside"].append(card)  # type: ignore
        else:
            self.players[player].discard.append(card)
        self.phase = TurnPhase.ACTION
        self._clear_pending()
        self._library_continue()
        return True

    def start_sentry(self, player: int) -> None:
        seen: List[CardDef] = []
        for _ in range(2):
            c = self._draw_one(player)
            if c:
                seen.append(c)

        if not seen:
            return

        self._sentry_ctx = {
            "player": player,
            "seen": seen,
            "idx": 0,
            "trash": [],
            "discard": [],
            "keep": [],
            "ordered": [],
        }
        self._sentry_next()

    def _sentry_next(self) -> None:
        if self._sentry_ctx is None:
            return
        if self.pending is not None:
            return

        player = int(self._sentry_ctx["player"])
        seen: List[CardDef] = self._sentry_ctx["seen"]  # type: ignore
        idx = int(self._sentry_ctx["idx"])

        if idx >= len(seen):
            keep: List[CardDef] = self._sentry_ctx["keep"]  # type: ignore
            trash_list: List[CardDef] = self._sentry_ctx["trash"]  # type: ignore
            discard_list: List[CardDef] = self._sentry_ctx["discard"]  # type: ignore

            self.trash.extend(trash_list)
            self.players[player].discard.extend(discard_list)

            if len(keep) <= 1:
                for c in reversed(keep):
                    self.players[player].deck.append(c)
                self._sentry_ctx = None
                self.phase = TurnPhase.ACTION
                return

            self.phase = TurnPhase.SENTRY_ORDER
            self.pending = Pending(TurnPhase.SENTRY_ORDER, player, {})
            return

        card = seen[idx]
        self.phase = TurnPhase.SENTRY_DECIDE_ONE
        self.pending = Pending(TurnPhase.SENTRY_DECIDE_ONE, player, {"card": card})

    def sentry_choose_trash(self) -> bool:
        return self._sentry_choose("trash")

    def sentry_choose_discard(self) -> bool:
        return self._sentry_choose("discard")

    def sentry_choose_keep(self) -> bool:
        return self._sentry_choose("keep")

    def _sentry_choose(self, bucket: str) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.SENTRY_DECIDE_ONE:
            return False
        if self._sentry_ctx is None:
            return False

        card = self.pending.data.get("card")
        if not isinstance(card, CardDef):
            return False

        self._sentry_ctx[bucket].append(card)  # type: ignore
        self._sentry_ctx["idx"] = int(self._sentry_ctx["idx"]) + 1  # type: ignore

        self.phase = TurnPhase.ACTION
        self._clear_pending()
        self._sentry_next()
        return True

    def sentry_order_choose(self, keep_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.SENTRY_ORDER:
            return False
        if self._sentry_ctx is None:
            return False
        player = self.pending.player
        keep: List[CardDef] = self._sentry_ctx["keep"]  # type: ignore
        ordered: List[CardDef] = self._sentry_ctx["ordered"]  # type: ignore

        if not (0 <= keep_index < len(keep)):
            return False

        ordered.append(keep.pop(keep_index))

        if len(keep) == 1:
            ordered.append(keep.pop(0))

        if len(keep) == 0 and len(ordered) > 0:
            for c in reversed(ordered):
                self.players[player].deck.append(c)

            self._sentry_ctx = None
            self.phase = TurnPhase.ACTION
            self._clear_pending()
            return True

        return True

    def start_artisan(self, player: int) -> None:
        self.phase = TurnPhase.ARTISAN_GAIN
        self.pending = Pending(TurnPhase.ARTISAN_GAIN, player, {"max_cost": 5})

    def artisan_gain(self, card_name: str) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.ARTISAN_GAIN:
            return False
        player = self.pending.player
        max_cost = int(self.pending.data.get("max_cost", 0))
        card = self.cards.get(card_name)
        if not card or card.cost > max_cost:
            return False

        ok = self.gain(player, card_name, where="hand")
        if not ok:
            return False

        self.phase = TurnPhase.ARTISAN_TOPDECK
        self.pending = Pending(TurnPhase.ARTISAN_TOPDECK, player, {})
        return True

    def artisan_gain_skip(self) -> None:
        if self.pending and self.pending.kind == TurnPhase.ARTISAN_GAIN:
            player = self.pending.player
            self.phase = TurnPhase.ARTISAN_TOPDECK
            self.pending = Pending(TurnPhase.ARTISAN_TOPDECK, player, {})

    def artisan_topdeck(self, hand_index: int) -> bool:
        if not self.pending or self.pending.kind != TurnPhase.ARTISAN_TOPDECK:
            return False
        player = self.pending.player
        p = self.players[player]
        if not p.hand:
            self.phase = TurnPhase.ACTION
            self._clear_pending()
            return True
        if not (0 <= hand_index < len(p.hand)):
            return False
        card = p.hand.pop(hand_index)
        p.deck.append(card)
        self.phase = TurnPhase.ACTION
        self._clear_pending()
        return True

    # -------------------- Scoring --------------------

    def compute_scores(self) -> Dict[int, int]:
        scores: Dict[int, int] = {}
        for idx, p in enumerate(self.players):
            all_cards = p.deck + p.hand + p.discard + p.in_play
            total = len(all_cards)
            score = 0
            score += sum(1 for c in all_cards if c.name == "Estate")
            score += 3 * sum(1 for c in all_cards if c.name == "Duchy")
            score += 6 * sum(1 for c in all_cards if c.name == "Province")
            score -= sum(1 for c in all_cards if c.name == "Curse")
            gardens = sum(1 for c in all_cards if c.name == "Gardens")
            score += gardens * (total // 10)
            scores[idx] = score
        return scores


# -------------------- Card definitions (Base 2E) --------------------


def create_card_defs() -> Dict[str, CardDef]:
    C = CardType
    cards: Dict[str, CardDef] = {}

    # Basic treasures
    cards["Copper"] = CardDef("Copper", 0, (C.TREASURE,), plus_coins=1)
    cards["Silver"] = CardDef("Silver", 3, (C.TREASURE,), plus_coins=2)
    cards["Gold"] = CardDef("Gold", 6, (C.TREASURE,), plus_coins=3)

    # Basic victory / curse
    cards["Estate"] = CardDef("Estate", 2, (C.VICTORY,))
    cards["Duchy"] = CardDef("Duchy", 5, (C.VICTORY,))
    cards["Province"] = CardDef("Province", 8, (C.VICTORY,))
    cards["Curse"] = CardDef("Curse", 0, (C.CURSE,))

    # ---- Base 2E Kingdom ----

    def cellar_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_cellar(player)

    cards["Cellar"] = CardDef(
        "Cellar", 2, (C.ACTION,), plus_actions=1, on_play=cellar_on_play
    )

    def chapel_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_chapel(player)

    cards["Chapel"] = CardDef("Chapel", 2, (C.ACTION,), on_play=chapel_on_play)

    cards["Moat"] = CardDef("Moat", 2, (C.ACTION, C.REACTION), plus_cards=2)

    def harbinger_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_harbinger(player)

    cards["Harbinger"] = CardDef(
        "Harbinger", 3, (C.ACTION,), plus_cards=1, plus_actions=1, on_play=harbinger_on_play
    )

    def merchant_on_play(game: Game, player: int, card: CardDef) -> None:
        game.merchant_bonus()

    cards["Merchant"] = CardDef(
        "Merchant", 3, (C.ACTION,), plus_cards=1, plus_actions=1, on_play=merchant_on_play
    )

    def vassal_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_vassal(player)

    cards["Vassal"] = CardDef(
        "Vassal", 3, (C.ACTION,), plus_coins=2, on_play=vassal_on_play
    )

    cards["Village"] = CardDef("Village", 3, (C.ACTION,), plus_cards=1, plus_actions=2)

    def workshop_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_workshop(player)

    cards["Workshop"] = CardDef("Workshop", 3, (C.ACTION,), on_play=workshop_on_play)

    def bureaucrat_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_bureaucrat(player)

    cards["Bureaucrat"] = CardDef(
        "Bureaucrat", 4, (C.ACTION, C.ATTACK), on_play=bureaucrat_on_play
    )

    cards["Gardens"] = CardDef("Gardens", 4, (C.VICTORY,))

    def militia_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_militia(player)

    cards["Militia"] = CardDef(
        "Militia", 4, (C.ACTION, C.ATTACK), plus_coins=2, on_play=militia_on_play
    )

    def moneylender_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_moneylender(player)

    cards["Moneylender"] = CardDef("Moneylender", 4, (C.ACTION,), on_play=moneylender_on_play)

    def poacher_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_poacher(player)

    cards["Poacher"] = CardDef(
        "Poacher", 4, (C.ACTION,), plus_cards=1, plus_actions=1, plus_coins=1, on_play=poacher_on_play
    )

    def remodel_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_remodel(player)

    cards["Remodel"] = CardDef("Remodel", 4, (C.ACTION,), on_play=remodel_on_play)

    cards["Smithy"] = CardDef("Smithy", 4, (C.ACTION,), plus_cards=3)

    def throne_room_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_throne_room(player)

    cards["Throne Room"] = CardDef(
        "Throne Room", 4, (C.ACTION,), on_play=throne_room_on_play
    )

    def bandit_on_play(game: Game, player: int, card: CardDef) -> None:
        game.gain(player, "Gold")

        for v in game.other_players(player):
            if game.player_has_moat(v):
                continue
            vp = game.players[v]
            revealed: List[CardDef] = []
            for _ in range(2):
                c2 = game._draw_one(v)
                if c2:
                    revealed.append(c2)
            if not revealed:
                continue

            trash_candidate = next(
                (c for c in revealed if c.has_type(C.TREASURE) and c.name != "Copper"),
                None
            )
            for c2 in revealed:
                if c2 is trash_candidate:
                    game.trash.append(c2)
                else:
                    vp.discard.append(c2)

    cards["Bandit"] = CardDef("Bandit", 5, (C.ACTION, C.ATTACK), on_play=bandit_on_play)

    def council_room_on_play(game: Game, player: int, card: CardDef) -> None:
        for v in game.other_players(player):
            game._draw_cards(v, 1)

    cards["Council Room"] = CardDef(
        "Council Room", 5, (C.ACTION,), plus_cards=4, plus_buys=1, on_play=council_room_on_play
    )

    cards["Festival"] = CardDef("Festival", 5, (C.ACTION,), plus_actions=2, plus_buys=1, plus_coins=2)

    cards["Laboratory"] = CardDef("Laboratory", 5, (C.ACTION,), plus_cards=2, plus_actions=1)

    def library_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_library(player)

    cards["Library"] = CardDef("Library", 5, (C.ACTION,), on_play=library_on_play)

    cards["Market"] = CardDef(
        "Market", 5, (C.ACTION,), plus_cards=1, plus_actions=1, plus_buys=1, plus_coins=1
    )

    def mine_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_mine(player)

    cards["Mine"] = CardDef("Mine", 5, (C.ACTION,), on_play=mine_on_play)

    def sentry_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_sentry(player)

    cards["Sentry"] = CardDef(
        "Sentry", 5, (C.ACTION,), plus_cards=1, plus_actions=1, on_play=sentry_on_play
    )

    def witch_on_play(game: Game, player: int, card: CardDef) -> None:
        for v in game.other_players(player):
            if game.player_has_moat(v):
                continue
            game.gain(v, "Curse")

    cards["Witch"] = CardDef(
        "Witch", 5, (C.ACTION, C.ATTACK), plus_cards=2, on_play=witch_on_play
    )

    def artisan_on_play(game: Game, player: int, card: CardDef) -> None:
        game.start_artisan(player)

    cards["Artisan"] = CardDef("Artisan", 6, (C.ACTION,), on_play=artisan_on_play)

    return cards


# Stable card name list for adapters / fixed action spaces
ALL_CARDS: List[str] = sorted(list(create_card_defs().keys()))
