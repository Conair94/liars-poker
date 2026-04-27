# Liar's Poker Agent — Literature Survey

A comprehensive review of literature supporting the R-NaD-based agent design for card-based Liar's Poker with test-time compute augmentation.

---

## 1. R-NaD Theory and Applications

### Perolat et al. 2022 — Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning
- **Venue/ID:** Science; arXiv:2206.15378
- **Summary:** Introduces DeepNash and Regularized Nash Dynamics (R-NaD), a multiagent RL algorithm that converges to approximate Nash equilibrium in imperfect-information games without search. Applies the method to Stratego, a game with ~10^535 game tree nodes and roughly 100-move episodes, demonstrating superhuman play (>84% win rate vs. expert humans) and top-3 all-time ranking on the Gravon platform. R-NaD combines entropy regularization of the utility function with policy iteration, iterating between self-play trajectory collection and policy gradient updates under a transformed reward that includes a regularization term penalizing deviation from an anchor policy.
- **Relevance to project:** Foundational R-NaD reference. Establishes convergence theory and empirical validation for large imperfect-information games with long episodes and many actions — directly applicable to card-based Liar's Poker. The regularization path interpretation (equilibrium traces out a path to Nash as regularization temperature changes) justifies the Stage 1 warm-start strategy. Architecture patterns (separate policy/value heads, transformer encoders for variable-length histories) inform the network design in §5.4.

### Dewey et al. 2025 — Outbidding and Outbluffing Elite Humans: Mastering Liar's Poker via Self-Play and Reinforcement Learning
- **Venue/ID:** arXiv:2511.03724 (November 2025)
- **Summary:** Applies R-NaD to Liar's Poker (dice variant, not card). Develops Solly, an agent trained via self-play actor-critic RL that achieves elite human-level performance (>50% win rate, superior equity metrics) in both heads-up and multiplayer formats, outperforming large language models. Provides hyperparameter ranges, network architecture details (policy + value heads, hand encoding, bid history), and evaluation methodology (exploitability, head-to-head win rates vs. human players). **Critically, does not employ test-time search** — this project extends with search-augmented inference.
- **Relevance to project:** Direct methodological precedent. Provides OpenSpiel game integration template, R-NaD hyperparameter ranges validated on a Liar's Poker variant, evaluation benchmarks, and architectural insights specific to bidding games. The dice variant differs structurally from the card variant (cards are non-exchangeable due to suit structure; dice are exchangeable), motivating investigation of test-time compute for non-exchangeable public states.

### Heinrich & Silver 2016 — Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP)
- **Venue/ID:** arXiv:1603.01121
- **Summary:** Introduces Neural Fictitious Self-Play (NFSP), combining fictitious self-play game-theoretic machinery with deep RL. Maintains two networks: a Q-network trained via off-policy RL (best-response approximator) and a π-network trained via supervised learning (average strategy). Convergence toward Nash equilibrium in Leduc poker; competitive performance with superhuman algorithms in Limit Texas Hold'em without domain knowledge.
- **Relevance to project:** NFSP is a foundational baseline. While R-NaD replaces NFSP's separate best-response/average-strategy networks with a unified policy under entropy regularization, understanding NFSP's convergence properties and sample efficiency is essential context. NFSP's success on Leduc suggests the architecture family (transformer + neural policy/value) is appropriate for card games at this scale.

---

## 2. Alternative Equilibrium Solvers

### Brown et al. 2019 — Deep Counterfactual Regret Minimization
- **Venue/ID:** ICML 2019; arXiv:1811.00164
- **Summary:** Introduces Deep CFR, applying neural networks to approximate CFR in full (unabstracted) games, circumventing the chicken-and-egg problem of manual game abstraction. Neural networks learn a regret approximator end-to-end. Achieves strong performance in large poker variants without domain-specific abstraction.
- **Relevance to project:** CFR is the canonical imperfect-information equilibrium solver (baseline for comparison). Deep CFR demonstrates that neural network function approximation of regret is viable at scale. Understanding Deep CFR's convergence path provides context for exploitability evaluation.

### Steinberger et al. 2020 — DREAM: Deep Regret Minimization with Advantage Baselines and Model-free Learning
- **Venue/ID:** NeurIPS 2020; arXiv:2006.10410
- **Summary:** Model-free variant of Deep CFR that does not require access to a perfect game simulator during training. Uses advantage baselines to reduce variance. Convergence guarantees to Nash equilibrium (two-player zero-sum). Two orders of magnitude better sample efficiency than NFSP on Leduc Hold'em.
- **Relevance to project:** DREAM addresses the simulator-free constraint (we use OpenSpiel's self-play rollouts). DREAM's advantage-baseline variance reduction inspired R-NaD's reward transformation strategy. A key baseline for sample efficiency comparison.

### McAleer et al. 2022 — ESCHER: Eschewing Importance Sampling in Games by Computing a History Value Function to Estimate Regret
- **Venue/ID:** ICML 2022; arXiv:2206.04122
- **Summary:** Improves on DREAM by eliminating importance sampling in regret estimation, replacing it with a learned history value function. Achieves orders-of-magnitude variance reduction. Outperforms DREAM and NFSP in large imperfect-information games. Unbiased, model-free, converges to approximate Nash equilibrium.
- **Relevance to project:** State-of-the-art variance-reduced CFR-style method. If iterative best-response is employed for test-time compute (§6), ESCHER's techniques could improve search-time performance.

### McAleer et al. 2021 — XDO: A Double Oracle Algorithm for Extensive-Form Games
- **Venue/ID:** NeurIPS 2021; arXiv:2103.06426
- **Summary:** Double-oracle algorithm that mixes best responses at every infostate, enabling linear convergence. Extends to neural best-response oracle (NXDO) for high-dimensional games. Achieves lower exploitability than CFR with the same computation in Leduc poker.
- **Relevance to project:** XDO/NXDO are competitive baselines. Could be an alternative if R-NaD convergence is slow at N=2.

---

## 3. Test-Time Compute for Imperfect-Information Games

### Brown et al. 2020 — Combining Deep Reinforcement Learning and Search for Imperfect-Information Games (ReBeL)
- **Venue/ID:** NeurIPS 2020; arXiv:2007.13544
- **Summary:** First algorithm to combine deep RL and search in two-player zero-sum imperfect-information games with provable convergence to Nash equilibrium. Generalizes AlphaZero to imperfect information via public belief states (PBS). Uses iterative gradient ascent on belief-space subgames. Achieves superhuman performance in heads-up no-limit Texas Hold'em without domain knowledge. Key insight: search leverages convexity of two-player zero-sum belief-space problems.
- **Relevance to project:** Most principled test-time compute method for our setting. The PBS formalism and subgame solving machinery are directly applicable to card Liar's Poker. **Critical open question from §9:** How expensive is belief propagation for non-exchangeable card states vs. exchangeable dice?

### Sokota et al. 2023 — Abstracting Imperfect Information Away from Two-Player Zero-Sum Games
- **Venue/ID:** ICML 2023; arXiv:2301.09159
- **Summary:** Shows that imperfect information can be abstracted away by publicly announcing policies at each infostate, provided one computes regularized (not unregularized) Nash equilibria. The announced-policy approach fails for plain Nash but works for regularized equilibria.
- **Relevance to project:** Justifies using regularized equilibria (which R-NaD produces) for test-time compute: ReBeL-style subgame decomposition is theoretically sound for regularized Nash even when public belief states miss non-exchangeable card details. Suggests iterative best-response on regularized subgames is principled.

### Brown et al. 2017 — Safe and Nested Subgame Solving for Imperfect-Information Games
- **Venue/ID:** NeurIPS 2017; arXiv:1705.02955
- **Summary:** Theory of subgame solving in imperfect-information games. Develops safe subgame resolving: given a pre-computed blueprint strategy, compute a refined strategy for the current subgame without losing performance guarantees. Nested subgame solving used by Libratus.
- **Relevance to project:** Foundational test-time search technique. The "safe" guarantees ensure search improves or maintains exploitability. The multi-round elimination structure naturally decomposes into subgames per round.

### Moravčík et al. 2017 — Libratus: The Superhuman AI for No-Limit Poker
- **Venue/ID:** Science 2017; IJCAI 2017
- **Summary:** First AI to defeat top human specialists in heads-up no-limit Texas Hold'em. Core: abstract the game, solve via CFR for a blueprint, then at runtime solve subgames to much higher granularity. Uses nested subgame solving with regularization toward the blueprint opponent model.
- **Relevance to project:** Libratus's pipeline (blueprint + runtime subgame solving) is directly applicable. Demonstrates subgame solving is the dominant test-time strategy in practice for large imperfect-information games.

---

## 4. Multi-Agent Generalization Beyond 2-Player Zero-Sum

### Farina et al. 2020 — Coarse Correlation in Extensive-Form Games
- **Venue/ID:** AAAI 2020; arXiv:1908.09893
- **Summary:** Extends regret minimization from two-player zero-sum to n-player general-sum games. Shows CFR-style learning converges to extensive-form coarse correlated equilibrium (EFCCE), weaker than Nash but still normative.
- **Relevance to project:** Justifies applying R-NaD to N > 2 players. For N ≥ 3, the appropriate solution concept is EFCCE, not Nash. Report empirical exploitability approximations and head-to-head win rates rather than formal Nash distance.

### Bai et al. 2022 — Near-Optimal Learning of Extensive-Form Games with Imperfect Information
- **Venue/ID:** ICML 2022
- **Summary:** Sample complexity bounds for learning approximate Nash equilibria in imperfect-information extensive-form games via self-play. Near-optimal iteration and sample complexity under natural structural assumptions.
- **Relevance to project:** Theoretical bounds inform convergence speed evaluation. Establishes asymptotic scalability of regret-based methods in our game class.

### Shi et al. 2022 — Optimal Policy of Multiplayer Poker via Actor-Critic Reinforcement Learning
- **Venue/ID:** *Entropy* 2022, 24(6), 774; MDPI; PMC9222241
- **Summary:** Applies Actor-Critic RL to 6-player Texas Hold'em using Centralized Training, Decentralized Execution (CTDE): Critic network sees all players' cards during training (perfect information) while Actor network operates on partial observations at execution. Proposes two multi-policy training schemes: **APU** (Asynchronous Policy Update — one agent updates per step, others frozen, to avoid simultaneous-update instability) and **Dual-APU** (all agents share parameters; an online network trains while a frozen target network plays against it, periodically synced). APU outperforms baselines (Sklansky heuristic, MCTS) by ~4 small bets/hand. LSTM over action history proved critical for extracting dynamic public information.
- **Relevance to project:** Three direct design implications:
  1. **APU for N>2 stability.** When training at N=3..5 (Stage C), apply APU: at each outer iteration, designate one seat as the learner while others play under a frozen snapshot of the policy. This prevents the oscillation/instability that arises when all N policies shift simultaneously — a known failure mode the paper explicitly addresses. R-NaD's anchor policy π_reg provides a natural mechanism: treat π_reg as the frozen "other agents" and update only a single per-seat copy per step.
  2. **Confirms EFCCE over Nash for N>2.** The paper explicitly acknowledges that Nash equilibrium is neither the right target nor theoretically validatable for N≥3 multiplayer poker, and reports results solely via win-rate comparison. This validates our §5.3 decision to restrict formal exploitability claims to N=2 and report win rates for N=3..5.
  3. **CTDE architecture.** The Critic-with-perfect-info architecture is a viable alternative to our single-head value network. If the single-network design underfits at N>2, CTDE (oracle Critic trained only against perfect-info targets) is the natural upgrade path. Note the Dewey et al. dice-variant agent did not use CTDE and still achieved elite performance; this paper's architecture is a contingency, not a required change.
- **Limitations:** No test-time adaptation or search; self-play only; cannot exploit specific opponent styles. Motivates M5.

---

## 5. Curriculum & Warm-Starting for Self-Play

### Ash et al. 2020 — On Warm-Starting Neural Network Training
- **Venue/ID:** NeurIPS 2020; arXiv:1910.08475
- **Summary:** Empirical study of warm-starting vs. cold-starting. Shows warm-starting reduces wall-clock time but often degrades generalization due to gradient imbalance. Proposes balanced warm-start techniques.
- **Relevance to project:** Our warm-start strategy uses **fixed auxiliary features** (Stage 1 probability tables), not network initialization — sidestepping the gradient imbalance problem. This paper suggests avoiding large prior weights that could bias early training, informing the auxiliary loss weight open question (§9).

### Supervised Auxiliary Tasks in Reinforcement Learning (Jaderberg et al., Caruana et al.)
- **Venue/ID:** Miscellaneous (longstanding line of work)
- **Summary:** Auxiliary tasks provide denser learning signal than the main reward — predicting opponent action, estimating state properties, etc. Benefits: faster convergence, better representation learning, improved sample efficiency.
- **Relevance to project:** The auxiliary loss that predicts pool hand distribution given private cards (§5.4) is directly inspired by this paradigm. Stage 1 conditional probability tables become supervised targets. The claim that auxiliary loss speeds convergence should be validated during M3–M4 training.

---

## 6. Liar's Poker Specifically

### Dewey et al. 2025 (see §1)
The only major AI/RL paper on Liar's Poker. Uses the dice variant; no prior work on the card variant.

### Wu & Wu 2024 — Exact Formulas for Poker Hand Probabilities in N-Card Games
- **Venue/ID:** arXiv:2309.00011
- **Summary:** Closed-form combinatorial counts for poker hand probabilities in n-card games. Covers Straight, Flush, Full House, Four of a Kind.
- **Relevance to project:** Stage 1 validation data. The "contains-at-least" counts are reference values for MC sanity checks; these are NOT interchangeable with "best-hand-equals" counts (see CLAUDE.md).

### General Notes
Card-based Liar's Poker is **essentially unstudied** in the academic RL/game-theory literature. The Dewey et al. paper is the first major AI work, and it uses the dice variant. The card variant is combinatorially richer (suit structure breaks exchangeability) and the multi-round elimination structure adds a meta-game layer not present in fixed-round dice Liar's Poker.

---

## Cross-Cutting Insights

### Convergence
Recent work (arXiv:2408.00751 — "A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence") establishes policy gradient convergence guarantees via bidilated regularization, showing best-iterate convergence to regularized Nash equilibrium. R-NaD inherits convergence guarantees from replicator dynamics + no-regret learning.

### Belief State Tracking
Imperfect-information games require tracking belief distributions over opponent states. In 2p0s, the public belief state (PBS) is a sufficient statistic. PBS computable via Bayes' rule over public observations. Sokota et al. (2023) show regularized Nash equilibria allow decision-time planning on belief space without correspondence issues.

---

---

## 7. Opponent Modeling and Range Estimation

This section covers literature newly relevant under the **exact-rules** variant of card Liar's Poker, where bids must be matched by a specific 5-card subset of the pool. Under exact rules, the strategic problem shifts from pure combinatorial threshold optimization (blackjack-like) toward opponent hand range reasoning (poker-like), making these references critical for the architecture design.

### Johanson, Zinkevich & Bowling 2007 — Computing Robust Counter-Strategies
- **Venue/ID:** NeurIPS 2007; https://poker.cs.ualberta.ca/publications/NIPS07-rnash.pdf
- **Summary:** Introduces the **Restricted Nash Response (RNR)** algorithm: given a suspected opponent tendency (an opponent model), compute a strategy that trades off between minimizing exploitability and maximizing exploitation. A single parameter p interpolates between the Nash equilibrium (p=1, fully safe) and the pure best response to the model (p=0, fully exploitative). Demonstrated in two-player Texas Hold'em. Builds on CFR as the underlying equilibrium solver.
- **Relevance to project:** Under exact rules, we may want to exploit observable opponent tendencies (e.g., overbidding certain hand types) while remaining robust. RNR formalizes exactly this tradeoff. The p parameter maps naturally onto the η regularization temperature in R-NaD: high η = near-Nash (robust), low η = near-best-response (exploitative). When we add opponent modeling features to the network, RNR provides the theoretical basis for deciding how aggressively to use them.

### Johanson & Bowling 2009 — Data-Biased Robust Counter-Strategies
- **Venue/ID:** AISTATS 2009; https://poker.cs.ualberta.ca/publications/AISTATS09.pdf
- **Summary:** Extends RNR by learning the opponent model from observed play data rather than assuming a fixed prior. Uses a Dirichlet prior over opponent action frequencies and updates it via Bayesian inference from match history. The posterior opponent model drives a best-response computation constrained to remain near Nash. Evaluated in full Texas Hold'em with large-scale experiments.
- **Relevance to project:** In the multi-round elimination match, we observe several rounds of opponent play before the match ends. Data-biased RNR is the principled way to exploit this accumulated evidence about opponent tendencies (e.g., bluffing rate, preferred bid escalation patterns) while staying robust. The Dirichlet prior maps cleanly onto the bid history encoder's output — observable bid frequencies become the sufficient statistic for the posterior.

### Ganzfried & Sandholm 2015 — Endgame Solving in Large Imperfect-Information Games
- **Venue/ID:** AAAI Workshop on Computer Poker 2015 / AAMAS; https://www.cs.cmu.edu/~sandholm/Endgame_AAAI15_workshop_cr_1.pdf
- **Summary:** Improves endgame (subgame) solving by computing the **joint** distribution of opponent private hands leading into an endgame, rather than assuming independent marginals. The joint distribution is derived via Bayes' rule applied to the full history of public actions under the pre-computed blueprint strategy. This corrects a systematic bias in prior endgame solvers and improved performance in the Annual Computer Poker Competition.
- **Relevance to project:** Directly applicable to our exact-rules game. Under exact rules, a call resolves based on whether any 5-card subset of the opponents' hands completes the bid — so the *joint* distribution of opponent holdings matters, not just per-opponent marginals. When implementing the Bayesian range tracker (§5.4), the update rule must track the joint distribution over all opponents' hands conditioned on the full bid history, exactly as Ganzfried & Sandholm prescribe.

### Bard, Johanson et al. 2013 — Online Implicit Agent Modelling
- **Venue/ID:** AAMAS 2013; https://poker.cs.ualberta.ca/publications/AAMAS13-modelling.pdf
- **Summary:** Introduces **implicit opponent modeling**: instead of estimating a generative model of the opponent's strategy (explicit modeling), maintain a portfolio of pre-computed counter-strategies and use a bandit algorithm to select among them online. The portfolio is constructed offline via submodular optimization and exploits variance reduction at inference. Won the 2011 Annual Computer Poker Competition by >95% confidence margin.
- **Relevance to project:** Implicit modeling is a lightweight alternative to explicit Bayesian range tracking. If full range tracking is too expensive or unstable in training, a portfolio of R-NaD checkpoints (each trained against a different opponent archetype) selected by a bandit at inference time is a practical fallback. The approach also informs how to structure the M5 test-time compute stage: rather than a single search, maintain a portfolio of subgame policies selected online.

### Ganzfried & Chiswick 2019 — Most Important Fundamental Rule of Poker Strategy
- **Venue/ID:** arXiv:1906.09895; AAAI 2019 Workshop
- **Summary:** Derives the **Range Advantage (RA)** correction to Minimum Defense Frequency (MDF). Standard MDF = b/(b+P) gives the fraction of hands a defender must call to make bluffs break even. This paper shows the optimal defense frequency is MDF − 0.5·RA + 0.25 where RA ∈ [0,1] measures how much stronger the bettor's range is than the defender's. Learned via machine learning on large databases of solver solutions; outperforms raw MDF by >60%.
- **Relevance to project:** Under exact rules, **range advantage is the central strategic concept**: a player who knows their opponent holds low-value hands can bid aggressively (polarized range) because the opponent cannot construct a completing 5-card subset for high bids. The RA formula provides a closed-form target for the call/fold decision that can be used as (a) a warm-start auxiliary loss target — predict RA from bid history — and (b) a sanity check for trained agent call frequencies. This is the academic formalization of the "polarized vs. merged range" practitioner concept.

### Notes on Exact-Rules Game Semantics
No academic paper specifically compares equilibrium structure under "at-least" vs. "exact-subset" resolution in bidding games. The "exact bid" is noted as a folk-rule variant in Liar's Dice (e.g., Wikipedia; UCLA "Models for the Game of Liar's Dice" by Tom Ferguson) but not analyzed in a game-theoretic paper. **This makes the exact-rules analysis in this project a genuine research novelty** — no existing equilibrium benchmark exists and the blind equilibrium must be computed fresh. Note also that neither "at-least" nor "exact" rules are formally standardized across Liar's Poker variants; the game has no canonical rule set.

---

## Summary Table

| Topic | Primary Reference | Alternative |
|-------|-------------------|-------------|
| R-NaD Theory | Perolat et al. 2022 (Stratego) | Dewey et al. 2025 (Liar's Poker dice) |
| Deep CFR | Brown et al. 2019 | Steinberger et al. 2020 (DREAM) |
| Test-Time Search | Brown et al. 2020 (ReBeL) | Moravčík et al. 2017 (Libratus) |
| Multi-Agent Theory | Farina et al. 2020 (EFCCE) | Bai et al. 2022 (Sample complexity) |
| Multi-Policy Training (N>2) | Shi et al. 2022 (APU/Dual-APU) | — |
| Warm-Start | Ash et al. 2020 | Jaderberg et al. (Auxiliary tasks) |
| Belief States | ReBeL (Brown et al. 2020) | Sokota et al. 2023 |
| Liar's Poker | Dewey et al. 2025 | Wu & Wu 2024 (Combinatorics) |
| **Opponent Modeling** | **Johanson et al. 2007 (RNR)** | **Bard et al. 2013 (Implicit Modeling)** |
| **Range Estimation** | **Ganzfried & Sandholm 2015 (Joint dist.)** | **Ganzfried & Chiswick 2019 (RA+MDF)** |

---

## Key Novelties of This Project (per survey)

1. **First application of R-NaD to card-based Liar's Poker** (not dice variant).
2. **First equilibrium analysis of the exact-rules variant** — no prior academic work on exact-subset resolution in Liar's Poker or Liar's Dice; the exact-rules blind equilibrium must be computed from scratch.
3. **Test-time search for non-exchangeable public states** (cards break the symmetry dice have; belief propagation cost is an open question).
4. **Warm-start via tabulated conditional probability features** (Bayesian prior as fixed network input, not initialization — no published precedent).
5. **Multi-round elimination meta-game** (dynamic hand sizes 1→5 with elimination; not studied in prior Liar's Poker work).
6. **Opponent range tracking as a primary strategic feature** under exact rules — formalized using joint distribution updates (Ganzfried & Sandholm) and RA-adjusted call frequencies (Ganzfried & Chiswick).
