
Vicinal is a local, context-aware AI watchdog that monitors, evaluates, and safeguards how developers use AI in real systems.


# Vicinal — Your AI Guardian

**Vicinal is runtime security infrastructure for AI systems.**
It sits *adjacent* to your AI calls, observes them in context, and intervenes only when something is about to go wrong.

Not a copilot.
Not a wrapper.
Not an “AI productivity tool”.

Vicinal is a **guardian layer for developers who already use a lot of AI and know how fragile it is**.

---

## Why Vicinal Exists

Modern AI systems fail quietly.

They hallucinate.
They leak data.
They drift after silent model updates.
They behave differently in prod than in dev.
They get jailbroken through user input and abstraction layers.

Most teams notice **after** damage is done.

Vicinal exists to make AI failures:

* observable
* contextual
* policy-driven
* defensible

If you’re expecting AI to be reliable by default, this project is not for you.

---

## Core Idea

Every AI call looks like this:

```
Application → Prompt → Model → Output → Application
```

Vicinal inserts itself **beside**, not above:

```
Application → Prompt → [Vicinal] → Model → Output → [Vicinal] → Application
```

Two checkpoints:

1. **Pre-call evaluation** (is this call safe to make?)
2. **Post-call evaluation** (is this output safe to use *here*?)

Vicinal does not generate content.
It does not improve answers.
It **judges behavior against policy and context**.

---

## What “Context-Aware” Actually Means

Context is not just text.

Vicinal reasons over:

* model identity and parameters
* endpoint purpose
* user role
* environment (dev / staging / prod)
* historical behavior
* industry constraints
* orchestration layers (fan-out, retries, ensembling)

The same output can be:

* acceptable in one context
* catastrophic in another

Vicinal treats those cases differently — by design.

---

## What Vicinal Does

* Intercepts LLM calls at runtime
* Normalizes requests and responses across providers
* Applies **policy-first guardrails**
* Detects:

  * prompt injection
  * PII and secret leakage
  * schema violations
  * hallucination risks
  * behavior drift across time or models
* Produces deterministic verdicts:

  * `ALLOW`
  * `WARN`
  * `BLOCK`
  * `REDACT`
* Logs and audits without spying by default

Think **WAF / IDS**, but for LLM behavior.

---

## What Vicinal Explicitly Does *Not* Do

* ❌ It does not replace your LLM
* ❌ It does not “rewrite prompts intelligently”
* ❌ It does not chase AI safety buzzwords
* ❌ It does not rely on LLMs as authoritative judges
* ❌ It does not hide behavior behind abstractions

If you want vibes, look elsewhere.

---

## Architecture Philosophy

### Policy-first, not NLP-first

Rules and invariants come before machine learning.
ML assists decisions — it does not own them.

### Determinism where it matters

Security decisions must be:

* explainable
* reproducible
* auditable

LLMs are used **sparingly and quarantined**.

### Adapters isolate chaos

Vendor APIs change.
Orchestrators mutate prompts.
Vicinal normalizes everything into a canonical form before evaluation.

---

## Supported Integration Surfaces

* Direct model APIs (OpenAI, Anthropic, Gemini, local runtimes)
* Orchestration frameworks (e.g. LangChain, LlamaIndex)
* Backend services via SDK middleware (Python, Node)

Every downstream model call is treated as an independent security event — even when abstractions try to hide that.

---

## Project Status

This repository is **private and under active development**.

Current focus:

* Runtime interception
* Canonical schemas
* Policy engine
* Deterministic pre/post-call evaluation

Non-goals (for now):

* Dashboards
* Billing
* Compliance theater
* Marketing demos

---

## Design Ethos (Read This Before Contributing)

* Assume models are unreliable.
* Assume users will do unsafe things.
* Assume abstractions lie.
* Assume silence is more dangerous than failure.

Vicinal exists because *trusting AI by default is irresponsible*.

---

## License & Security

This project treats security as a first-class concern.
See `SECURITY.md` for disclosure practices.

---



You’re doing this the right way.
