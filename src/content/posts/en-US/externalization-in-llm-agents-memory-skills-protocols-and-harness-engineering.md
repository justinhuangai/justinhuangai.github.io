---
title: "Externalization in LLM Agents: Cognitive Artifacts for Agent Engineering"
description: "A reading of Externalization in LLM Agents through the lens of cognitive artifacts: agent progress is increasingly about moving memory, skills, protocols, and runtime governance outside the model."
date: "2026-05-13T16:00:00+08:00"
category: "Paper Reading"
tags: [paper-reading, agents, externalization, harness-engineering, memory, skills, protocols, LLM]
pinned: false
---

Most conversations about agents eventually drift back to model rankings.

Which model reasons better, which one has the larger context window, which one wins a coding benchmark. Those questions matter. But [“Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering”](/papers/2604.08224v1.pdf) points at a different center of gravity.

The paper is not mainly asking what else can be compressed into a model. It asks:

**Which cognitive burdens should no longer be carried by the model in the first place?**

That is why the key word is `externalization`. The paper is not just describing extra components around an LLM. It is describing a change in task representation. Human artifacts work this way too. A list does not enlarge biological memory. It turns recall into recognition. A map does not make the brain intrinsically better at navigation. It makes spatial relations visible.

The paper argues that serious LLM agents are moving through the same kind of representational shift.

![Figure 1 from the paper: externalization as the organizing principle of LLM agent design](/images/posts/externalization-in-llm-agents/figure-1-externalization-overview.png)

*Figure 1, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), format-converted for web display, content unchanged.*

## What this survey clarifies

This is not a new model paper. It is not a benchmark paper. Its value is conceptual: it gives a name to an engineering migration that is already happening.

At first, we treated capability as something mostly inside weights: larger models, better pretraining, better alignment. Then capability increasingly moved into context: prompts, few-shot examples, RAG, reasoning traces, tool descriptions. Now practical agent systems rely on a thicker runtime: memory stores, file systems, skill libraries, tool protocols, sandboxes, approval gates, logs, evaluators, and sub-agent orchestration.

The paper compresses that arc into:

**weights → context → harness**

![Author's conceptual map of cognitive externalization in LLM agents](/images/posts/externalization-in-llm-agents/cognitive-externalization-map.webp)

*AI-generated conceptual diagram by the author; not from the paper. It summarizes this article's reading of cognitive externalization: model capability moves from weights, through context, into harness-level infrastructure such as memory, skills, protocols, and governance.*

That framing is useful because it matches what real systems feel like. The model still matters, but it no longer carries the entire task alone. It operates inside an environment that remembers, retrieves, constrains, audits, and recovers.

Reliability, in this view, is not only a property of the model. It is also a property of how the task has been externalized.

## Memory externalizes time

Memory is the easiest form of externalization to understand.

If an agent only has a context window, every run depends on a temporary working surface. Longer context helps, but it does not remove the selection problem. What should be included? What should be forgotten? What is stale? What is noise? More importantly, a context window is ephemeral unless state is persisted somewhere else.

External memory turns the problem from “can the model recall this from weights or prompt residue?” into “can the model recognize and use the right retrieved state?” That is exactly the cognitive-artifact pattern. A shopping list does not make memory bigger. It changes the task.

For an agent, memory may store user preferences, project conventions, past decisions, failed attempts, domain facts, or recurring constraints. The hard part is not merely storage. The hard part is curation: what gets written, when it gets retrieved, how much is injected, how it is compressed, and how old state is prevented from contaminating new work.

So memory is not just a cheaper substitute for a bigger context window. It is infrastructure for continuity across time.

## Skills externalize procedure

The second form is skills.

A skill is not simply “the model can call a tool.” A skill is reusable procedural knowledge. It may contain steps, heuristics, stopping conditions, escalation rules, recovery patterns, and safety constraints. A mature skill tells the agent how this class of work should usually be done.

That is a different layer from tool use. Tools expose actions. Protocols define how actions are discovered and invoked. Skills encode how actions should be organized into a repeatable task.

![Figure 5 from the paper: skills as externalized procedural expertise](/images/posts/externalization-in-llm-agents/figure-5-skills-lifecycle.png)

*Figure 5, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), reproduced unchanged.*

Software agents make this especially concrete.

A model may know how to edit code, run tests, inspect logs, and write a pull request summary. Reliable execution still depends on local procedure: which files to read first, how to preserve user changes, when to use `rg`, when to run a full validation pass, how to handle an existing staging area, and which commands are too destructive to run casually.

If all of that is left for the model to rediscover in every session, behavior drifts. When it is externalized as a skill, the task changes from improvising a workflow to selecting and following a validated one.

That is why skills are easy to underestimate. They are not flashy, but they reduce variance.

## Protocols externalize interaction order

The third form is protocols.

Without protocols, agent interaction is mostly free-text negotiation. The model says it wants a tool call. A tool returns text. Another agent infers what that text means. This can work in demos, but it is brittle in production.

Protocols turn ambiguous interaction into machine-readable contracts. Tool discovery, argument schemas, permissions, errors, delegation, lifecycle management, and user approvals should not be held together only by prompt convention.

The value is not just interoperability. Protocols also create governance surfaces. Once an interaction is structured, the system can validate, audit, replay, monitor, and restrict it. Free text is flexible, but it is hard to govern.

## Harness is a cognitive environment

The paper’s strongest move is to place memory, skills, and protocols inside harness engineering.

A harness is not a thin wrapper around a model. It is the cognitive environment in which the agent runs: the control loop, context budget, permission model, sandbox, human approval path, logs, evaluation hooks, failure recovery, and sub-agent orchestration.

![Figure 3 from the paper: externalization architecture of a harnessed LLM agent](/images/posts/externalization-in-llm-agents/figure-3-harnessed-agent-architecture.png)

*Figure 3, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), reproduced unchanged.*

The important point is that the harness is not merely one more module next to memory, skills, and protocols. It is the runtime that hosts and coordinates them. Memory supplies state. Skills supply procedure. Protocols supply interaction structure. The harness decides when those structures are loaded, how they interact, and what constraints apply.

This is why mature agents increasingly resemble small operating environments. They have resources, permissions, lifecycle, logs, policies, and recovery paths. They are not just “a model plus tools.”

## The boundary of capability is moving

The most useful idea in the paper is that it changes the question “where is the capability?”

If capability lives only in weights, agent improvement mostly means using a better model, fine-tuning, or retraining. If capability also lives in context, prompts and retrieval become major engineering surfaces. If capability lives in the harness too, system design itself becomes part of capability.

This does not make the model less important. It makes strong models more worth surrounding with good infrastructure. LLMs are strong at synthesis, judgment, and generalization over provided information. They are not naturally reliable at persistent memory, repeatable procedure, permission management, long-lived state, or cross-system coordination.

Externalization is not cheating. It is engineering around real boundaries by changing the task.

A good agent system does not force the model to start from scratch every time. It turns durable state into memory, reusable procedure into skills, governable exchange into protocols, and runtime reliability into a harness.

## The cost of externalization

The framework is compelling, but externalization is not free.

Memory introduces stale state, privacy boundaries, and retrieval pollution. Skills can become outdated, overfit, or unsafe when composed incorrectly. Protocols can fragment into incompatible standards or lock systems into rigid interfaces. Harnesses increase complexity: the more approval gates, logs, sandboxes, policies, and subroutines a system has, the more engineering discipline it needs.

There is also an evaluation problem. If agent capability is distributed across the model and external infrastructure, what exactly are we measuring? The same model can behave very differently inside different harnesses. “Model capability” and “agent capability” are no longer the same thing.

That is the point worth keeping. The paper is not saying that externalization solves everything. It says reliable agency is a joint product of model and environment.

## My takeaway

The paper can be compressed into one practical sentence:

**Agent engineering is increasingly harness engineering.**

Not because models stopped mattering, but because models now operate inside a larger cognitive system. Memory extends them across time. Skills stabilize procedure. Protocols impose interaction order. Harnesses make the whole thing observable, permissioned, and recoverable.

When evaluating an agent system, I would therefore ask less “which model does it use?” and more:

- What state has it externalized?
- What reusable skills does it carry?
- Are its tools and agent interactions protocolized?
- How does its harness handle permissions, failures, logs, and human approval?

Those questions often reveal more about the real system than the model name does.
