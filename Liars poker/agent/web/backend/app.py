"""
FastAPI backend for Liar's Poker web interface (M6a).

Run from papers/Liars poker/:
    python agent/web/run.py
    # or
    uvicorn agent.web.backend.app:app --reload
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR   = os.path.abspath(os.path.join(_BACKEND_DIR, "..", ".."))
_PAPER_DIR   = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.engine import MatchState, RoundResult, new_match  # noqa: E402
from agent.game.bids import (  # noqa: E402
    CALL_ACTION, NUM_BIDS, all_bids, bid_to_index, index_to_bid,
    HAND_NAMES, RANK_NAMES,
)
from .agents import BlindBaselineAgent, ConditionalAgent, RandomAgent  # noqa: E402

app = FastAPI(title="Liar's Poker")

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
# session_id -> {
#   "state":                MatchState,
#   "human_seat":           int,
#   "agents":               {seat: Agent},
#   "last_result":          Optional[RoundResult],
#   "waiting_next_round":   bool,
#   "current_round_history": list[(seat, action)],  # accumulates during active round
#   "last_round_history":    list[(seat, action)],  # snapshot of completed round
# }
_SESSIONS: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Card / bid display helpers
# ---------------------------------------------------------------------------
_RANK_DISPLAY = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
_SUIT_SYMBOL  = ["♣", "♦", "♥", "♠"]
_SUIT_COLOR   = ["#222", "#c0392b", "#c0392b", "#222"]


def _card_html(card_idx: int) -> str:
    rank = card_idx >> 2
    suit = card_idx & 3
    color = _SUIT_COLOR[suit]
    sym   = _SUIT_SYMBOL[suit]
    return (
        f'<span class="card" style="color:{color}">'
        f'{_RANK_DISPLAY[rank]}{sym}'
        f'</span>'
    )


def _bid_label(bid_idx: int) -> str:
    b = index_to_bid(bid_idx)
    return f"{HAND_NAMES[b.hand_type]} {_RANK_DISPLAY[b.primary_rank]}"


def _bid_short(bid_idx: int) -> str:
    b = index_to_bid(bid_idx)
    abbrev = ["HC", "Pr", "2P", "3K", "St", "Fl", "FH", "4K", "SF"]
    return f"{abbrev[b.hand_type]}-{_RANK_DISPLAY[b.primary_rank]}"


# ---------------------------------------------------------------------------
# Auto-advance agents until it's the human's turn (or round / match ends)
# ---------------------------------------------------------------------------

def _advance_agents(session_id: str) -> None:
    """Apply agent actions until the human is to act, or the round / match ends."""
    sess = _SESSIONS[session_id]
    state: MatchState   = sess["state"]
    agents: dict        = sess["agents"]
    human_seat: int     = sess["human_seat"]

    while (
        not state.terminal
        and state.round_state is not None
        and state.round_state.current_player != human_seat
    ):
        cp = state.round_state.current_player
        if cp not in agents:
            break   # Shouldn't happen, but guard anyway

        action = agents[cp].choose_action(state)
        sess["current_round_history"].append((cp, action))
        result = state.apply_action(action)

        if result is not None:
            sess["last_result"]         = result
            sess["last_round_history"]  = list(sess["current_round_history"])
            sess["waiting_next_round"]  = True
            return

    sess["waiting_next_round"] = False


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    min-height: 100vh;
    padding: 1rem;
}
h1 { font-size: 1.8rem; color: #c9a84c; margin-bottom: 0.25rem; }
h2 { font-size: 1.1rem; color: #a0a8c0; margin-bottom: 0.75rem; font-weight: 500; }
h3 { font-size: 0.95rem; color: #8090a8; text-transform: uppercase;
     letter-spacing: 0.05em; margin-bottom: 0.5rem; }

.container { max-width: 900px; margin: 0 auto; }
.panel {
    background: #16213e;
    border: 1px solid #2a3a5e;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
label { display: block; margin-bottom: 0.3rem; font-size: 0.9rem; color: #a0a8c0; }
select, input {
    background: #0f3460;
    color: #e0e0e0;
    border: 1px solid #2a4a7e;
    border-radius: 4px;
    padding: 0.4rem 0.7rem;
    font-size: 0.9rem;
    width: 100%;
    margin-bottom: 0.75rem;
}
button, .btn {
    background: #c9a84c;
    color: #1a1a2e;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1.2rem;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
}
button:hover, .btn:hover { background: #e0bf6a; }
button.danger { background: #c0392b; color: #fff; }
button.danger:hover { background: #e74c3c; }
button.secondary { background: #2a3a5e; color: #c0d0f0; }
button.secondary:hover { background: #3a4e7e; }

.card {
    display: inline-block;
    background: #fff;
    border: 1px solid #bbb;
    border-radius: 4px;
    padding: 0.15rem 0.4rem;
    font-size: 1rem;
    font-weight: 700;
    margin: 0.1rem;
    min-width: 2rem;
    text-align: center;
}
.cards-row { display: flex; flex-wrap: wrap; gap: 0.2rem; align-items: center; }

.player-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e2e4e;
}
.player-row:last-child { border-bottom: none; }
.player-name { min-width: 110px; font-size: 0.9rem; }
.player-name.active-player { color: #c9a84c; font-weight: 600; }
.player-name.human { color: #5ba3d0; }
.player-name.eliminated { color: #555; text-decoration: line-through; }
.hand-dots { display: flex; gap: 3px; }
.hand-dot { width: 14px; height: 20px; background: #2a4a7e; border-radius: 2px; }
.hand-dot.used { background: #c9a84c; }
.badge {
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    background: #2a3a5e;
    color: #a0b0d0;
}
.badge.turn { background: #c9a84c22; color: #c9a84c; border: 1px solid #c9a84c; }

.bid-history { list-style: none; font-size: 0.85rem; color: #8090a8; max-height: 120px; overflow-y: auto; }
.bid-history li { padding: 0.15rem 0; border-bottom: 1px solid #1e2e4e22; }
.bid-history li.current-bid { color: #c9a84c; font-weight: 600; }

.action-area { margin-top: 0.5rem; }
.action-area form { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: flex-end; }
.action-area .form-group { flex: 1; min-width: 200px; }
.action-area select { margin-bottom: 0; }

.result-box {
    background: #0f3460;
    border: 1px solid #2a5a9e;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.result-box .pool-cards { margin: 0.5rem 0; }
.result-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; }
.result-title.win  { color: #27ae60; }
.result-title.lose { color: #c0392b; }
.result-title.info { color: #c9a84c; }

.winner-banner {
    background: linear-gradient(135deg, #c9a84c22, #27ae6022);
    border: 2px solid #c9a84c;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
}
.winner-banner h2 { font-size: 1.5rem; color: #c9a84c; }
.winner-banner p  { color: #a0c0a0; margin-top: 0.5rem; }

.status-msg { color: #8090a8; font-size: 0.9rem; font-style: italic; }

.htmx-indicator { display: none; }
.htmx-request .htmx-indicator { display: inline; }

details.rules-panel summary { cursor: pointer; list-style: none; }
details.rules-panel summary::-webkit-details-marker { display: none; }
details.rules-panel summary h3 { display: inline-block; margin-bottom: 0; }
details.rules-panel summary h3::after { content: " ▸"; }
details[open].rules-panel summary h3::after { content: " ▾"; }
.rules-content { margin-top: 0.8rem; font-size: 0.88rem; line-height: 1.6; color: #b0c0d8; }
.rules-content h4 { color: #c9a84c; margin: 0.8rem 0 0.3rem; font-size: 0.9rem; }
.rules-content ul { padding-left: 1.2rem; }
.rules-content li { margin-bottom: 0.2rem; }
.rules-content .highlight { color: #e0e0e0; font-weight: 600; }

.bid-select-group { display: flex; gap: 0.5rem; align-items: flex-end; flex: 1; }
.bid-select-group select { flex: 1; margin-bottom: 0; }
optgroup { background: #0d2a50; color: #8090a8; font-size: 0.8rem; }
"""


def _render_full_page(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Liar's Poker</title>
  <script src="https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"></script>
  <style>{_BASE_CSS}</style>
</head>
<body>
  <div class="container">
    <h1>Liar's Poker</h1>
    <h2>Card-Based Variant — Nash Equilibrium Research</h2>
    {body_html}
  </div>
</body>
</html>"""


_RULES_HTML = """
<details class="panel rules-panel">
  <summary><h3>Game Rules</h3></summary>
  <div class="rules-content">
    <h4>Objective</h4>
    <ul>
      <li>Last player standing wins. Players are eliminated when they reach 5 cards.</li>
    </ul>
    <h4>Each Round</h4>
    <ul>
      <li>Every active player is dealt fresh private cards (face-down). You start with <span class="highlight">1 card</span>; the count grows when you lose rounds.</li>
      <li>The designated first bidder names a <span class="highlight">poker hand</span> (e.g. "Pair 9", "Straight A"). Each subsequent player must name a <span class="highlight">strictly stronger hand</span> or <span class="highlight">Call Bluff</span>.</li>
      <li>The first bidder of the round must bid — calling is not allowed before any bid exists.</li>
    </ul>
    <h4>Calling Bluff</h4>
    <ul>
      <li>When you call, all hands are revealed and combined into a single <span class="highlight">pool</span>.</li>
      <li>The pool's <span class="highlight">best 5-card poker hand</span> is evaluated.</li>
      <li>If pool ≥ standing bid → <span class="highlight">caller loses</span> (the bid was truthful).</li>
      <li>If pool &lt; standing bid → <span class="highlight">bidder loses</span> (it was a bluff).</li>
    </ul>
    <h4>After the Round</h4>
    <ul>
      <li>Loser gets +1 card. If loser already had 5 cards → <span class="highlight">eliminated</span>.</li>
      <li>Loser becomes the first bidder of the next round (or next active seat if eliminated).</li>
    </ul>
    <h4>Hand Rankings (weakest → strongest)</h4>
    <ul>
      <li>High Card → Pair → Two Pair → Three of a Kind → Straight → Flush → Full House → Four of a Kind → Straight Flush (Ace-high = Royal Flush)</li>
      <li>Within each type, higher primary rank wins (e.g. Pair A beats Pair K).</li>
    </ul>
    <h4>Agents</h4>
    <ul>
      <li><span class="highlight">Random</span> — picks any legal action uniformly at random.</li>
      <li><span class="highlight">Blind Baseline</span> — uses the N=2 backward-induction equilibrium (ignores its private cards; calls at the ~50% probability threshold).</li>
      <li><span class="highlight">Conditional</span> — uses private-card conditional probabilities from the pre-computed tables to set a hand-informed threshold bid.</li>
    </ul>
  </div>
</details>
"""


def _render_setup_form() -> str:
    return _RULES_HTML + """
<div class="panel" id="setup-panel">
  <h3>New Game</h3>
  <form hx-post="/game/new" hx-target="#game-area" hx-swap="outerHTML"
        hx-indicator="#spinner">
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
      <div>
        <label>Number of Players</label>
        <select name="num_players">
          <option value="2" selected>2</option>
          <option value="3">3</option>
          <option value="4">4</option>
          <option value="5">5</option>
        </select>
      </div>
      <div>
        <label>Your Seat</label>
        <select name="human_seat">
          <option value="-1">Spectator (watch agents)</option>
          <option value="0" selected>Seat 0 (first bidder)</option>
          <option value="1">Seat 1</option>
          <option value="2">Seat 2</option>
          <option value="3">Seat 3</option>
          <option value="4">Seat 4</option>
        </select>
      </div>
      <div>
        <label>Opponent</label>
        <select name="agent_type">
          <option value="random">Random</option>
          <option value="blind" selected>Blind Baseline (N=2 equilibrium)</option>
          <option value="conditional">Conditional (private-card priors)</option>
        </select>
      </div>
    </div>
    <button type="submit">Start Game</button>
    <span id="spinner" class="htmx-indicator status-msg"> Loading...</span>
  </form>
</div>
<div id="game-area"></div>
"""


def _player_label(seat: int, human_seat: int) -> str:
    return "You" if seat == human_seat else f"Agent {seat}"


def _render_players(state: MatchState, human_seat: int) -> str:
    cp = state.round_state.current_player if state.round_state else -1
    rows = []
    for seat in range(state.num_players):
        eliminated = not state.active[seat]
        is_human = (seat == human_seat)
        name_cls = "player-name"
        if eliminated:
            name_cls += " eliminated"
        elif is_human:
            name_cls += " human"
        if seat == cp:
            name_cls += " active-player"

        label = _player_label(seat, human_seat)
        size  = state.hand_sizes[seat]

        dots = "".join(
            f'<div class="hand-dot{"" if i >= size else " used"}"></div>'
            for i in range(5)
        )

        badge = ""
        if seat == cp and not eliminated:
            badge = '<span class="badge turn">▶ Acting</span>'
        elif eliminated:
            badge = '<span class="badge">Eliminated</span>'

        rows.append(f"""
        <div class="player-row">
          <div class="{name_cls}">{label}</div>
          <div class="hand-dots">{dots}</div>
          <div style="font-size:0.8rem;color:#8090a8;">{size} card{'s' if size != 1 else ''}</div>
          {badge}
        </div>""")

    return (
        '<div class="panel">'
        '<h3>Players</h3>'
        + "".join(rows)
        + "</div>"
    )


def _render_hand(own_hand: list, session_id: str) -> str:
    cards_html = " ".join(_card_html(c) for c in sorted(own_hand))
    return f"""
<div class="panel">
  <h3>Your Hand</h3>
  <div class="cards-row">{cards_html}</div>
</div>"""


def _render_all_hands(state: MatchState) -> str:
    """Spectator view: show all active players' face-up hands."""
    rs = state.round_state
    rows = []
    for seat in range(state.num_players):
        if not state.active[seat]:
            continue
        hand = sorted(rs.hands[seat]) if rs else []
        cards_html = " ".join(_card_html(c) for c in hand) if hand else "(no cards)"
        rows.append(
            f'<div style="margin-bottom:0.5rem;">'
            f'<span style="color:#8090a8;font-size:0.85rem;min-width:70px;display:inline-block;">'
            f'Agent {seat}</span> '
            f'<span class="cards-row" style="display:inline-flex;">{cards_html}</span>'
            f'</div>'
        )
    return (
        '<div class="panel">'
        '<h3>Hands (spectator view)</h3>'
        + "".join(rows)
        + "</div>"
    )


def _render_bid_history(state: MatchState, human_seat: int) -> str:
    rs = state.round_state
    if rs is None:
        return ""

    bids_list = all_bids()
    items = []
    for turn_seat, action in rs.history:
        who = _player_label(turn_seat, human_seat)
        if action == CALL_ACTION:
            items.append(f"<li><b>{who}</b>: CALL</li>")
        else:
            items.append(f'<li class="current-bid"><b>{who}</b>: {_bid_label(action)}</li>')

    hist_html = "".join(items) if items else "<li style='color:#555'>No bids yet</li>"

    cur_bid_str = "None"
    if rs.current_bid is not None:
        cur_bid_str = f"{HAND_NAMES[rs.current_bid.hand_type]} {_RANK_DISPLAY[rs.current_bid.primary_rank]}"

    return f"""
<div class="panel">
  <h3>Round — Standing Bid: <span style="color:#c9a84c">{cur_bid_str}</span></h3>
  <ul class="bid-history">{hist_html}</ul>
</div>"""


def _render_action_area(state: MatchState, session_id: str) -> str:
    legal     = state.legal_actions()
    can_call  = CALL_ACTION in legal
    bid_idxs  = [a for a in legal if a != CALL_ACTION]

    # Group legal bids by hand type into <optgroup> sections.
    # This lets users navigate "Pair → rank" instead of scrolling 100 flat entries.
    if bid_idxs:
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for idx in bid_idxs:
            b = index_to_bid(idx)
            groups[b.hand_type].append((idx, b))

        option_html = ""
        for ht in range(9):   # HC=0 … SF=8
            if ht not in groups:
                continue
            option_html += f'<optgroup label="{HAND_NAMES[ht]}">'
            for idx, b in groups[ht]:
                option_html += (
                    f'<option value="{idx}">'
                    f'{_RANK_DISPLAY[b.primary_rank]}</option>'
                )
            option_html += '</optgroup>'

        bid_form_html = f"""
    <form hx-post="/game/{session_id}/action" hx-target="#game-area" hx-swap="outerHTML"
          style="display:flex; gap:0.5rem; align-items:flex-end; flex-wrap:wrap; flex:1;">
      <div style="flex:1; min-width:180px;">
        <label style="margin-bottom:0.2rem;">Hand type &amp; rank</label>
        <select name="action">{option_html}</select>
      </div>
      <button type="submit" style="margin-bottom:0.75rem;">Raise ↑</button>
    </form>"""
    else:
        bid_form_html = ""

    call_btn = (
        f'<form hx-post="/game/{session_id}/action" hx-target="#game-area" hx-swap="outerHTML">'
        f'<input type="hidden" name="action" value="{CALL_ACTION}">'
        f'<button type="submit" class="danger" style="margin-top:1.55rem;">Call Bluff</button>'
        f'</form>'
    ) if can_call else ""

    return f"""
<div class="panel action-area">
  <h3>Your Action</h3>
  <div style="display:flex; gap:0.75rem; flex-wrap:wrap; align-items:flex-start;">
    {bid_form_html}
    {call_btn}
  </div>
</div>"""


def _render_round_history(history: list, human_seat: int) -> str:
    """Display the sequence of bids/calls made during the completed round."""
    if not history:
        return ""
    items = []
    for seat, action in history:
        who = _player_label(seat, human_seat)
        if action == CALL_ACTION:
            items.append(f'<li><b>{who}</b>: <span style="color:#c0392b">Call Bluff</span></li>')
        else:
            items.append(f'<li><b>{who}</b>: {_bid_label(action)}</li>')
    return (
        '<div class="panel" style="margin-top:0.5rem">'
        '<h3>Round Sequence</h3>'
        '<ul class="bid-history" style="max-height:200px">'
        + "".join(items)
        + "</ul></div>"
    )


def _render_result(result: RoundResult, human_seat: int, state: MatchState, session_id: str, history: list | None = None) -> str:
    pool_html = " ".join(_card_html(c) for c in sorted(result.pool))
    pool_best_str = (
        f"{HAND_NAMES[result.pool_best.hand_type]} "
        f"{_RANK_DISPLAY[result.pool_best.primary_rank]}"
    )
    bid_str = (
        f"{HAND_NAMES[result.bid.hand_type]} "
        f"{_RANK_DISPLAY[result.bid.primary_rank]}"
    )

    caller_name = _player_label(result.caller_seat, human_seat)
    loser_name  = _player_label(result.loser_seat, human_seat)

    if result.call_succeeded:
        outcome = f"{caller_name} called the bluff — pool was only {pool_best_str} (bid was {bid_str})!"
    else:
        outcome = f"{caller_name}'s call failed — pool had {pool_best_str} ≥ {bid_str}!"

    if result.loser_seat == human_seat:
        title_cls  = "result-title lose"
        title_text = "You lost this round — +1 card penalty"
    elif result.winner_seat == human_seat:
        title_cls  = "result-title win"
        title_text = "You won this round!"
    else:
        title_cls  = "result-title info"
        title_text = f"{loser_name} lost this round"

    # Determine next-round or game-over button
    if state.terminal:
        action_btn = ""
    else:
        action_btn = f"""
        <form hx-post="/game/{session_id}/next-round" hx-target="#game-area" hx-swap="outerHTML"
              style="margin-top:0.75rem">
          <button type="submit">Next Round →</button>
        </form>"""

    return f"""
<div class="result-box">
  <div class="{title_cls}">{title_text}</div>
  <div style="margin:0.4rem 0; font-size:0.9rem">{outcome}</div>
  <div class="pool-cards">
    <span style="color:#8090a8; font-size:0.85rem;">Pool: </span>
    {pool_html}
  </div>
</div>
{_render_round_history(history or [], human_seat)}
{action_btn}"""


def _render_game_area(session_id: str) -> str:
    sess  = _SESSIONS[session_id]
    state: MatchState = sess["state"]
    human_seat: int   = sess["human_seat"]
    spectator         = (human_seat == -1)
    last_result       = sess.get("last_result")
    waiting_next      = sess.get("waiting_next_round", False)
    last_history      = sess.get("last_round_history", [])

    new_game_btn = """
        <form hx-get="/setup" hx-target="body" hx-swap="innerHTML" style="margin-top:1rem">
          <button type="submit">New Game</button>
        </form>"""

    # ---------- terminal ----------
    if state.terminal:
        winner_name = _player_label(state.winner, human_seat)
        if spectator:
            banner_msg = f"{winner_name} won the match."
            banner_sub = "Match over."
        else:
            you_won    = (state.winner == human_seat)
            banner_msg = "You won the match!" if you_won else f"{winner_name} won the match."
            banner_sub = "Congratulations — you beat the opponent!" if you_won else "Better luck next time."

        result_block = ""
        if last_result is not None:
            result_block = _render_result(last_result, human_seat, state, session_id, last_history)

        return f"""
<div id="game-area">
  {result_block}
  <div class="winner-banner">
    <h2>{banner_msg}</h2>
    <p>{banner_sub}</p>
    {new_game_btn}
  </div>
</div>"""

    # ---------- between rounds ----------
    if waiting_next and last_result is not None:
        result_block  = _render_result(last_result, human_seat, state, session_id, last_history)
        players_block = _render_players(state, human_seat)
        return f"""
<div id="game-area">
  {players_block}
  {result_block}
</div>"""

    # ---------- active round (spectator) ----------
    # In spectator mode the entire round has already been played by _advance_agents;
    # we should never reach here — but guard just in case.
    rs = state.round_state
    if rs is None:
        return f'<div id="game-area"><p class="status-msg">Starting round...</p></div>'

    if spectator:
        return f"""
<div id="game-area">
  {_render_players(state, human_seat)}
  {_render_all_hands(state)}
  {_render_bid_history(state, human_seat)}
  <div class="panel"><p class="status-msg">Round in progress...</p></div>
</div>"""

    # ---------- active round (player) ----------
    players_block  = _render_players(state, human_seat)
    hand_block     = _render_hand(rs.hands[human_seat], session_id)
    history_block  = _render_bid_history(state, human_seat)

    is_human_turn = (rs.current_player == human_seat)

    if is_human_turn:
        action_block = _render_action_area(state, session_id)
    else:
        acting_name  = _player_label(rs.current_player, human_seat)
        action_block = f'<div class="panel"><p class="status-msg">{acting_name} is thinking...</p></div>'

    return f"""
<div id="game-area">
  {players_block}
  {hand_block}
  {history_block}
  {action_block}
</div>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    body = _render_setup_form()
    return HTMLResponse(_render_full_page(body))


@app.get("/setup", response_class=HTMLResponse)
def setup_fragment():
    """Return the full page (used after game ends for 'New Game' button)."""
    body = _render_setup_form()
    return HTMLResponse(_render_full_page(body))


@app.post("/game/new", response_class=HTMLResponse)
def new_game(
    num_players: int  = Form(2),
    human_seat:  int  = Form(0),
    agent_type:  str  = Form("random"),
):
    # -1 = spectator; otherwise clamp to valid seat range
    if human_seat != -1:
        human_seat = max(0, min(human_seat, num_players - 1))

    session_id = uuid.uuid4().hex[:8]
    state = new_match(num_players)

    agents = {}
    for seat in range(num_players):
        if seat != human_seat:  # human_seat == -1 fills every seat
            if agent_type == "blind":
                agents[seat] = BlindBaselineAgent()
            elif agent_type == "conditional":
                agents[seat] = ConditionalAgent()
            else:
                agents[seat] = RandomAgent()

    _SESSIONS[session_id] = {
        "state":                  state,
        "human_seat":             human_seat,
        "agents":                 agents,
        "last_result":            None,
        "waiting_next_round":     False,
        "current_round_history":  [],
        "last_round_history":     [],
    }

    # Start the first round.
    state.start_next_round()

    # If agents go before the human, auto-advance.
    _advance_agents(session_id)

    return HTMLResponse(_render_game_area(session_id))


@app.post("/game/{session_id}/action", response_class=HTMLResponse)
def take_action(session_id: str, action: int = Form(...)):
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    sess  = _SESSIONS[session_id]
    state = sess["state"]

    if state.terminal:
        return HTMLResponse(_render_game_area(session_id))

    if state.round_state is None:
        raise HTTPException(status_code=400, detail="No active round")

    if action not in state.legal_actions():
        raise HTTPException(status_code=400, detail=f"Illegal action: {action}")

    sess["current_round_history"].append((state.round_state.current_player, action))
    result = state.apply_action(action)
    if result is not None:
        sess["last_result"]        = result
        sess["last_round_history"] = list(sess["current_round_history"])
        sess["waiting_next_round"] = True
        return HTMLResponse(_render_game_area(session_id))

    # Human bid — advance agents.
    _advance_agents(session_id)
    return HTMLResponse(_render_game_area(session_id))


@app.post("/game/{session_id}/next-round", response_class=HTMLResponse)
def next_round(session_id: str):
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    sess  = _SESSIONS[session_id]
    state = sess["state"]

    if state.terminal:
        return HTMLResponse(_render_game_area(session_id))

    sess["last_result"]              = None
    sess["waiting_next_round"]       = False
    sess["current_round_history"]    = []

    state.start_next_round()
    _advance_agents(session_id)

    return HTMLResponse(_render_game_area(session_id))
