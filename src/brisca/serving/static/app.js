"use strict";

const SUITS = {
  O: { name: "Oros", symbol: "🪙", color: "var(--oros)" },
  C: { name: "Copas", symbol: "🏆", color: "var(--copas)" },
  E: { name: "Espadas", symbol: "⚔️", color: "var(--espadas)" },
  B: { name: "Bastos", symbol: "🪵", color: "var(--bastos)" },
};
const RANK_NAMES = { 1: "As", 10: "Sota", 11: "Caballo", 12: "Rey" };
const AGENT_LABELS = {
  random: "Random",
  greedy: "Greedy (one move ahead)",
  heuristic: "Heuristic (hand-written rules)",
  ismcts: "ISMCTS (search, strongest)",
  alphabeta: "Alpha-beta (search)",
  ppo: "PPO (reinforcement learning)",
};

const $ = (id) => document.getElementById(id);
let game = null;
let busy = false;
let suggested = null;

function parseCard(text) {
  return { rank: Number(text.slice(0, -1)), suit: text.slice(-1) };
}

function cardElement(text, { small = false, onClick = null } = {}) {
  const { rank, suit } = parseCard(text);
  const info = SUITS[suit];
  const el = document.createElement(onClick ? "button" : "div");
  el.className = small ? "card small" : "card";
  el.style.setProperty("--suit", info.color);
  el.setAttribute("aria-label", `${RANK_NAMES[rank] || rank} de ${info.name}`);
  el.innerHTML =
    `<span class="rank">${rank}</span>` +
    `<span class="name">${RANK_NAMES[rank] || ""}</span>` +
    `<span class="suit" aria-hidden="true">${info.symbol}</span>`;
  if (onClick) {
    el.type = "button";
    el.addEventListener("click", onClick);
  }
  return el;
}

function fill(container, cards, options) {
  container.replaceChildren(...cards.map((c) => cardElement(c, options)));
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

function render() {
  if (!game) return;
  const opponent = game.agent;
  $("opponent-name").textContent = opponent;
  $("score-you").textContent = game.scores.you;
  $("score-ai").textContent = game.scores[opponent];
  fill($("trump"), [game.trump_card], { small: true });
  $("stock").textContent = game.trump_drawn ? "deck empty" : `${game.stock_size} in deck`;
  fill($("trick"), game.current_trick);

  const last = game.last_trick;
  fill($("last-trick"), last ? last.cards : []);
  $("last-trick-note").textContent = last
    ? `${last.winner === "you" ? "You" : opponent} won ${last.points} point${last.points === 1 ? "" : "s"}`
    : "";

  const hand = $("hand");
  hand.classList.toggle("disabled", !game.your_turn || busy);
  hand.replaceChildren(
    ...game.hand.map((c) => {
      const el = cardElement(c, { onClick: game.your_turn && !busy ? () => play(c) : null });
      el.classList.toggle("suggested", c === suggested);
      return el;
    })
  );
  $("hint").disabled = !game.your_turn || busy;

  if (game.finished) {
    const verdict = { win: "You win!", loss: `${opponent} wins.`, draw: "A 60-60 draw." }[game.result];
    $("message").textContent = `${verdict} Final score ${game.scores.you}-${game.scores[opponent]}.`;
  } else if (busy) {
    $("message").textContent = `${opponent} is thinking...`;
  } else {
    $("message").textContent = game.current_trick.length
      ? `${opponent} led. Your move.`
      : "Your lead.";
  }
}

function cardLabel(text) {
  const { rank, suit } = parseCard(text);
  return `${RANK_NAMES[rank] || rank} de ${SUITS[suit].name}`;
}

async function showHint() {
  const panel = $("hint-panel");
  panel.hidden = false;
  panel.textContent = "Analysing the position...";
  $("hint").disabled = true;
  try {
    const hint = await api(`/v1/games/${game.game_id}/hint`, {});
    suggested = hint.card;
    const odds = hint.moves
      .map((m) => `${cardLabel(m.card)} ${Math.round(100 * m.win_chance)}%`)
      .join(" · ");
    panel.replaceChildren();
    const chances = hint.moves.map((m) => m.win_chance);
    // The engine recommends its most-explored move; when estimates are this close,
    // the ranking is within its margin of error, so say so.
    const close = Math.max(...chances) - Math.min(...chances) < 0.05;
    const lead = document.createElement("p");
    lead.textContent = `Suggested: ${cardLabel(hint.card)}.` +
      (close ? " It's a close call: every option gives similar chances." : "");
    panel.append(lead);
    if (hint.explanation) {
      const why = document.createElement("p");
      why.textContent = hint.explanation;
      panel.append(why);
    }
    const detail = document.createElement("p");
    detail.className = "odds";
    detail.textContent = `Estimated chance of winning the game: ${odds}`;
    panel.append(detail);
  } catch (error) {
    panel.textContent = error.message;
  } finally {
    render();
  }
}

function clearHint() {
  suggested = null;
  $("hint-panel").hidden = true;
}

async function play(card) {
  clearHint();
  busy = true;
  render();
  try {
    game = await api(`/v1/games/${game.game_id}/moves`, { card });
  } catch (error) {
    $("message").textContent = error.message;
  } finally {
    busy = false;
    render();
  }
}

async function newGame() {
  clearHint();
  busy = true;
  $("new-game").disabled = true;
  try {
    game = await api("/v1/games", { agent: $("agent").value });
  } catch (error) {
    $("message").textContent = error.message;
  } finally {
    busy = false;
    $("new-game").disabled = false;
    render();
  }
}

async function init() {
  const agents = await api("/v1/agents");
  $("agent").replaceChildren(
    ...agents.map((a) => new Option(AGENT_LABELS[a.id] || a.id, a.id, false, a.id === "ismcts"))
  );
  $("new-game").addEventListener("click", newGame);
  $("hint").addEventListener("click", showHint);
  await newGame();
}

init().catch((error) => {
  $("message").textContent = `Could not reach the API: ${error.message}`;
});
