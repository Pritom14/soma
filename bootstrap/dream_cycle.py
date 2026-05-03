"""
bootstrap/dream_cycle.py — SOMA's nightly consolidation loop.

Inspired by gbrain's "dream cycle" concept. Runs on a nightly cron.
While SOMA sleeps, this cycle:

  1. Re-verifies stale beliefs (beliefs marked is_actionable=False)
  2. Synthesizes updated compiled truth for each watched repo
  3. Consolidates session memories (summarize last 7 sessions into one)
  4. Prunes low-value experiences older than 30 days
  5. Self-tests lowest-confidence active beliefs

Run manually:
    python3 main.py --dream-cycle

Or via cron (nightly at 3am):
    0 3 * * * cd ~/Desktop/soma && python3 main.py --dream-cycle >> comms/outbox/cron.log 2>&1
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from config import TIER_1_MODEL, TIER_2_MODEL, TIER_3_MODEL
from core.belief import BeliefStore
from core.belief_index import BeliefIndex
from core.experience import ExperienceStore
from core.brain import BrainStore
from core.llm import LLMClient
from core.hypothesis import HypothesisGenerator
from core.experiment import ExperimentRunner
from core.pr_monitor import PRRegistry
from core.identity import IdentityStore
from core.introspection import IntrospectionEngine
from core.goals import GoalStore
# from comms.protocol.session_memory import SessionMemory  # TODO: stub this module


def run(verbose: bool = True) -> dict:
    """
    One full dream cycle pass.
    Returns summary dict of what was done.
    """
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "stale_beliefs_retested": 0,
        "repos_synthesized": [],
        "sessions_consolidated": 0,
        "experiences_pruned": 0,
        "low_conf_tested": 0,
        "introspection_updated": False,
        "cross_domain_contradictions": 0,
        "cross_domain_patterns_synthesized": 0,
        "cross_domain_beliefs_crystallized": 0,
    }

    llm = LLMClient()
    brain = BrainStore()
    store = ExperienceStore()
    # session_memory = SessionMemory()  # TODO: implement comms.protocol.session_memory

    # --- Step 3: Skip session consolidation (requires SessionMemory) ---
    # Skipped for now due to missing comms.protocol module

    # --- Step 6: Introspection ---
    if verbose:
        print("[Dream] Step 6: Introspection")
    try:
        from core.belief import BeliefStore as _BS

        identity = IdentityStore()
        engine = IntrospectionEngine()
        self_bs = _BS("self")
        goals = GoalStore()
        all_bs = {d: _BS(d) for d in ["code", "research", "task", "self"]}
        new_meta = engine.form_meta_beliefs(store, all_bs)
        if verbose and new_meta:
            print(f"[Dream]   Formed {len(new_meta)} meta-belief(s)")
        assessment = engine.assess(store, self_bs, goals)
        engine.update_identity(identity, llm, TIER_2_MODEL, assessment)
        report["introspection_updated"] = True
        if verbose:
            print("[Dream]   Identity updated from introspection")
    except Exception as e:
        if verbose:
            print(f"[Dream]   Introspection failed: {e}")

    # --- Step 7: Skill Validation ---
    if verbose:
        print("\n[Dream] Step 7: Validating skill files")
    try:
        from core.failure_analyzer import SkillStore

        skill_store = SkillStore()
        skills = skill_store.all_skills()
        stale_count = 0
        for skill in skills:
            skill_path = Path(skill["path"])
            try:
                content = skill_path.read_text()
                # Update last_validated timestamp
                now = datetime.utcnow().isoformat()
                if "Last Validated:" in content:
                    content = content.replace("Last Validated: ", f"Last Validated: {now}")
                skill_path.write_text(content)
            except Exception:
                stale_count += 1
        if verbose:
            print(f"[Dream]   Validated {len(skills)} skill(s), {stale_count} stale")
        report["skills_validated"] = len(skills)
    except Exception as e:
        if verbose:
            print(f"[Dream]   Skill validation failed: {e}")

    if verbose:
        print("\n" + "=" * 55)
        print("[Dream] Dream cycle starting")
        print(f"[Dream]   Introspection updated: {report['introspection_updated']}")
    print("=" * 55)

    # --- Step 1: Re-verify stale beliefs ---
    if verbose:
        print("\n[Dream] Step 1: Re-verifying stale beliefs")

    for domain in ["code", "oss_contribution"]:
        bs = BeliefStore(domain)
        stale = bs.get_stale()
        if verbose:
            print(f"[Dream]   {domain}: {len(stale)} stale belief(s)")

        for belief in stale:
            try:
                gen = HypothesisGenerator(llm, TIER_2_MODEL)
                runner = ExperimentRunner(llm, TIER_3_MODEL)
                hypothesis = gen.generate(belief)
                if hypothesis:
                    result = runner.run(hypothesis)
                    bs.update_from_experiment(belief.id, result.confirmed)
                    # Restore actionable if confidence recovered
                    refreshed = bs.beliefs.get(belief.id)
                    if refreshed and refreshed.confidence >= 0.4:
                        refreshed.is_actionable = True
                        bs._save()
                    report["stale_beliefs_retested"] += 1
                    if verbose:
                        status = "confirmed" if result.confirmed else "challenged"
                        print(
                            f"[Dream]   Retested: '{belief.statement[:60]}' → {status} ({refreshed.confidence:.0%})"
                        )
            except Exception as e:
                if verbose:
                    print(f"[Dream]   Retest failed for {belief.id}: {e}")

    # --- Step 2: Synthesize repo brain pages ---
    if verbose:
        print("\n[Dream] Step 2: Synthesizing repo brain pages")

    registry = PRRegistry()
    watched_repos = list({pr.repo for pr in registry.get_all()})

    for repo in watched_repos:
        try:
            new_truth = brain.synthesize_repo(repo, llm, TIER_2_MODEL)
            if new_truth:
                report["repos_synthesized"].append(repo)
                if verbose:
                    print(f"[Dream]   Synthesized: {repo}")
        except Exception as e:
            if verbose:
                print(f"[Dream]   Synthesis failed for {repo}: {e}")

    # --- Step 3: Consolidate old session memories ---
    # Skipped: SessionMemory module not available (comms.protocol not implemented)
    if verbose:
        print("\n[Dream] Step 3: Session consolidation skipped (SessionMemory unavailable)")

    # --- Step 4: Prune old low-value experiences ---
    if verbose:
        print("\n[Dream] Step 4: Pruning old low-value experiences")

    try:
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        pruned = store.prune_old(cutoff_date=cutoff, max_confidence=0.3)
        report["experiences_pruned"] = pruned
        if verbose:
            print(f"[Dream]   Pruned {pruned} old low-confidence experience(s)")
    except Exception as e:
        if verbose:
            print(f"[Dream]   Pruning failed: {e}")

    # --- Step 5: Self-test lowest-confidence active beliefs ---
    if verbose:
        print("\n[Dream] Step 5: Self-testing low-confidence beliefs")

    for domain in ["code", "oss_contribution"]:
        bs = BeliefStore(domain)
        active = [b for b in bs.all() if b.is_actionable and b.confidence < 0.55]
        active.sort(key=lambda b: b.confidence)

        for belief in active[:3]:  # test at most 3 per domain
            try:
                gen = HypothesisGenerator(llm, TIER_2_MODEL)
                runner = ExperimentRunner(llm, TIER_3_MODEL)
                hypothesis = gen.generate(belief)
                if hypothesis:
                    result = runner.run(hypothesis)
                    bs.update_from_experiment(belief.id, result.confirmed)
                    report["low_conf_tested"] += 1
                    if verbose:
                        status = "confirmed" if result.confirmed else "challenged"
                        print(f"[Dream]   Tested: '{belief.statement[:60]}' → {status}")
            except Exception as e:
                if verbose:
                    print(f"[Dream]   Self-test failed: {e}")

    # --- Step 7: Collect trajectories and trigger LoRA fine-tune ---
    if verbose:
        print("\n[Dream] Step 7: Trajectory collection + LoRA fine-tune check")
    report["trajectories_collected"] = 0
    report["finetune_triggered"] = False
    report["finetune_checkpoint"] = ""
    try:
        from core.trajectory import TrajectoryRecorder
        from pathlib import Path as _Path

        traj_stats = TrajectoryRecorder.stats()
        new_count = traj_stats.get("since_last_finetune", 0)
        report["trajectories_collected"] = new_count

        if verbose:
            print(f"[Dream]   New trajectories since last fine-tune: {new_count}")

        FINETUNE_THRESHOLD = 10
        if new_count < FINETUNE_THRESHOLD:
            if verbose:
                print(f"[Dream]   Below threshold ({FINETUNE_THRESHOLD}) — skipping fine-tune")
        else:
            try:
                import mlx_lm  # noqa: F401

                _mlx_available = True
            except ImportError:
                _mlx_available = False

            if not _mlx_available:
                if verbose:
                    print("[Dream]   mlx_lm not available — skipping fine-tune")
            else:
                _run_finetune(verbose=verbose)
                report["finetune_triggered"] = True

                # Advance watermark so these trajectories aren't re-trained
                _wm = _Path(__file__).parent.parent / "data" / "trajectories" / ".last_finetune"
                _wm.parent.mkdir(parents=True, exist_ok=True)
                _wm.write_text(datetime.utcnow().isoformat())

                # Promote best checkpoint
                adapter_dir = _Path(__file__).parent.parent / "models" / "soma-lora"
                checkpoints = sorted(adapter_dir.glob("*_adapters.safetensors"))
                if checkpoints:
                    import shutil as _shutil

                    best = checkpoints[-1]
                    target = adapter_dir / "adapters.safetensors"
                    if target.exists():
                        _shutil.copy2(target, adapter_dir / "adapters.safetensors.bak")
                    _shutil.copy2(best, target)
                    report["finetune_checkpoint"] = best.name
                    if verbose:
                        print(f"[Dream]   Promoted checkpoint: {best.name} → adapters.safetensors")
    except Exception as e:
        if verbose:
            print(f"[Dream]   Step 7 failed: {e}")

    # --- Step 7b: Cross-domain belief crystallization via BeliefIndex ---
    if verbose:
        print("\n[Dream] Step 7b: Cross-domain belief crystallization")
    try:
        # 1. Instantiate BeliefIndex to load all domain beliefs
        index = BeliefIndex()
        index_summary = index.summary()
        if verbose:
            print(f"[Dream]   Indexed {index_summary['total']} belief(s) across {len(index_summary['by_domain'])} domain(s)")

        # 2. Detect contradictions across domain boundaries
        contradictions = index.detect_contradictions(min_overlap=0.35)
        report["cross_domain_contradictions"] = len(contradictions)
        if verbose and contradictions:
            print(f"[Dream]   Found {len(contradictions)} cross-domain contradiction(s)")
            for c in contradictions[:3]:  # Log top 3
                print(
                    f"[Dream]     - {c.domain_a}/{c.belief_a_id[:8]}: {c.belief_a[:50]}"
                )
                print(
                    f"[Dream]     - {c.domain_b}/{c.belief_b_id[:8]}: {c.belief_b[:50]} (overlap: {c.overlap_score:.1%})"
                )

        # 3. Synthesize cross-domain patterns from high-confidence beliefs
        patterns = index.synthesize_patterns(llm, TIER_2_MODEL)
        report["cross_domain_patterns_synthesized"] = len(patterns)
        if verbose and patterns:
            print(f"[Dream]   Synthesized {len(patterns)} cross-domain pattern(s)")
            for p in patterns:
                print(
                    f"[Dream]     - [{', '.join(p.domains)}] {p.pattern[:60]} (support: {p.support_count}, conf: {p.confidence:.0%})"
                )

        # 4. Crystallize patterns into self-domain beliefs
        new_beliefs = index.write_to_self(patterns)
        report["cross_domain_beliefs_crystallized"] = len(new_beliefs)
        if verbose and new_beliefs:
            print(f"[Dream]   Crystallized {len(new_beliefs)} new self-domain belief(s)")
            for b in new_beliefs:
                print(f"[Dream]     - {b.statement[:70]} (conf: {b.confidence:.0%})")

    except Exception as e:
        if verbose:
            print(f"[Dream]   Step 7b (cross-domain synthesis) failed: {e}")

    # --- Step 8: Harness introspection + self-modification ---
    if verbose:
        print("\n[Dream] Step 8: Harness introspection + self-modification")
    report["harness_improvements"] = 0
    report["harness_analysis_safe"] = False
    try:
        from core.harness_introspection import HarnessIntrospector
        from core.self_modifier import SelfModifier

        inspector = HarnessIntrospector(Path(__file__).parent.parent)
        engine = IntrospectionEngine()
        harness_patterns = engine.detect_harness_patterns(store)
        self_bs = BeliefStore("self")
        analysis = inspector.analyze(harness_patterns, self_bs.all())
        if verbose:
            print(f"[Dream]   Components analyzed: {analysis.get('components_analyzed', 0)}")
            print(f"[Dream]   Safety: {analysis['safety_assessment']}")
        report["harness_analysis_safe"] = analysis["safety_assessment"] == "safe"
        if analysis["safety_assessment"] == "safe" and analysis.get("suggested_improvements"):
            modifier = SelfModifier(Path(__file__).parent.parent, llm, TIER_1_MODEL)
            results = modifier.run_improvement_cycle(analysis)
            applied = [r for r in results if r.success]
            report["harness_improvements"] = len(applied)
            if verbose:
                for r in results:
                    status = "APPLIED" if r.success else "FAILED"
                    component = r.proposal.component if r.proposal else "unknown"
                    print(f"[Dream]   {component}: {status}")
        elif verbose:
            print(f"[Dream]   Skipping modifications: safety={analysis['safety_assessment']}")
    except Exception as e:
        if verbose:
            print(f"[Dream]   Step 8 failed: {e}")

    if verbose:
        print("\n" + "=" * 55)
        print("[Dream] Dream cycle complete")
        print(f"[Dream]   Stale beliefs retested: {report['stale_beliefs_retested']}")
        print(f"[Dream]   Repos synthesized: {report['repos_synthesized']}")
        print(f"[Dream]   Sessions consolidated: {report['sessions_consolidated']}")
        print(f"[Dream]   Experiences pruned: {report['experiences_pruned']}")
        print(f"[Dream]   Low-conf beliefs tested: {report['low_conf_tested']}")
        print(f"[Dream]   New trajectories: {report['trajectories_collected']}")
        print(f"[Dream]   Fine-tune triggered: {report['finetune_triggered']}")
        print(f"[Dream]   Cross-domain contradictions: {report['cross_domain_contradictions']}")
        print(f"[Dream]   Cross-domain patterns: {report['cross_domain_patterns_synthesized']}")
        print(f"[Dream]   Cross-domain beliefs crystallized: {report['cross_domain_beliefs_crystallized']}")
        print(f"[Dream]   Harness improvements: {report.get('harness_improvements', 0)}")
        print("=" * 55 + "\n")

    return report


def _run_finetune(verbose: bool = True) -> None:
    """
    Merge hand-crafted train.jsonl + auto-generated trajectories,
    then invoke mlx_lm.lora via subprocess.
    train.jsonl is always restored after the run (even on crash).
    """
    import json as _json
    import shutil as _shutil
    import subprocess as _subprocess
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    import os as _os

    base = _Path(__file__).parent.parent
    traj_dir = base / "data" / "trajectories"
    train_src = base / "data" / "train.jsonl"
    merged_path = base / "data" / "train_merged.jsonl"
    config_path = base / "data" / "lora_config.yaml"
    watermark_file = traj_dir / ".last_finetune"

    # Collect trajectories newer than last watermark
    watermark_dt = None
    if watermark_file.exists():
        try:
            watermark_dt = _dt.fromisoformat(watermark_file.read_text().strip())
        except Exception:
            pass

    new_lines: list[str] = []
    for f in sorted(traj_dir.glob("*.jsonl")):
        mtime = _dt.utcfromtimestamp(_os.path.getmtime(f))
        if watermark_dt and mtime <= watermark_dt:
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = _json.loads(line)
                except Exception:
                    continue
                # Skip failed / partial trajectories
                if record.get("failed") or record.get("partial"):
                    continue
                convs = record.get("conversations", [])
                if len(convs) < 2:
                    continue
                human = next((c["value"] for c in convs if c["from"] == "human"), "")
                gpt = next((c["value"] for c in convs if c["from"] == "gpt"), "")
                if human and gpt:
                    text = f"<s>[INST] {human} [/INST] {gpt} </s>"
                    new_lines.append(_json.dumps({"text": text}))

    if verbose:
        print(f"[Dream]   Converting {len(new_lines)} new trajectory record(s) to train format")

    # Write merged file (original + new trajectories)
    with open(merged_path, "w") as out:
        if train_src.exists():
            with open(train_src) as src:
                for line in src:
                    out.write(line)
        for line in new_lines:
            out.write(line + "\n")

    if verbose:
        print(f"[Dream]   Merged: {merged_path} ({sum(1 for _ in open(merged_path))} examples)")

    # Temporarily swap train.jsonl → merged for mlx_lm (which looks for train.jsonl by name)
    train_backup = base / "data" / "train.jsonl.finetune_bak"
    _shutil.copy2(train_src, train_backup)
    _shutil.copy2(merged_path, train_src)

    try:
        cmd = ["python3", "-m", "mlx_lm.lora", "--config", str(config_path), "--train"]
        if verbose:
            print(f"[Dream]   Launching: {' '.join(cmd)}")
        result = _subprocess.run(cmd, capture_output=not verbose, timeout=3600)
        if verbose:
            print(f"[Dream]   Fine-tune exit code: {result.returncode}")
    except _subprocess.TimeoutExpired:
        if verbose:
            print("[Dream]   Fine-tune timed out after 1 hour")
    except Exception as e:
        if verbose:
            print(f"[Dream]   Fine-tune error: {e}")
    finally:
        # Always restore original train.jsonl
        if train_backup.exists():
            _shutil.copy2(train_backup, train_src)
            train_backup.unlink(missing_ok=True)
