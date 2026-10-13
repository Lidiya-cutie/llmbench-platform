#!/usr/bin/env python3
"""
Immediate Action Plan for Enterprise-Grade Model Comparison System

This script provides safe, non-disruptive steps to transform the current
system into a reliable, company-ready product.

PHASE 1: Foundation (SAFE - No process disruption)
PHASE 2: Integration (CAREFUL - Minimal interference)
PHASE 3: Enterprise (COMPLETE - Company ready)
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class EnterprisePlan:
    """Concrete action plan for enterprise transformation"""

    def __init__(self):
        self.base_dir = Path("/opt/llm-data/llmbench")
        self.state_dir = self.base_dir / ".llmbench" / "state"
        self.backup_dir = self.base_dir / ".llmbench" / "backups"

    def create_directories(self):
        """Create necessary directories for state management"""
        print("📁 Creating state management directories...")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = self.state_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        print("✅ Directories created")
        return True

    def save_current_state_snapshot(self):
        """Save snapshot of current model states without disruption"""
        print("💾 Creating system state snapshot...")

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self._get_system_status(),
            'models': self._get_all_model_states(),
            'active_processes': self._get_active_processes()
        }

        snapshot_file = self.state_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        snapshot_file.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))

        print(f"✅ Snapshot saved: {snapshot_file.name}")
        return snapshot_file

    def _get_system_status(self) -> Dict[str, Any]:
        """Check current system status without disruption"""
        import subprocess

        status = {
            'gpu_available': False,
            'active_models': 0,
            'system_load': 'unknown'
        }

        try:
            # Check if nvidia-smi is available
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                status['gpu_available'] = True
                status['active_models'] = len(result.stdout.strip().split('\n'))
        except:
            pass

        return status

    def _get_all_model_states(self) -> List[Dict[str, Any]]:
        """Get state of all models from cards"""
        models = []

        for card_file in self.base_dir.glob("out/card_*.json"):
            if ".bak." in card_file.name:
                continue

            try:
                with open(card_file) as f:
                    card = json.load(f)

                model = card.get('name', 'unknown')
                perf = card.get('perf', {})
                quality = card.get('quality', {})

                # Determine completeness
                is_complete = (
                    perf.get('ok', 0) > 0 and  # Has successful perf tests
                    len([k for k, v in quality.items() if v > 0]) >= 2  # At least 2 quality metrics
                )

                models.append({
                    'name': model,
                    'file': str(card_file),
                    'complete': is_complete,
                    'perf_ok': perf.get('ok', 0),
                    'perf_n': perf.get('n', 0),
                    'quality_metrics': len([k for k, v in quality.items() if v > 0]),
                    'missing_metrics': self._get_missing_metrics(card)
                })
            except Exception as e:
                print(f"  Warning: Could not read {card_file.name}: {e}")

        return sorted(models, key=lambda m: m['name'])

    def _get_missing_metrics(self, card: Dict[str, Any]) -> List[str]:
        """Identify missing metrics in model card"""
        missing = []

        # Check perf
        perf = card.get('perf', {})
        if perf.get('ok', 0) == 0 or perf.get('n', 0) == 0:
            missing.append('perf')

        # Check quality
        quality = card.get('quality', {})
        if quality.get('mmlu_pro', 0) == 0:
            missing.append('mmlu_pro')
        if quality.get('humaneval', 'MISSING') == 'MISSING' or quality.get('humaneval', 0) == 0:
            missing.append('humaneval')
        if quality.get('gsm8k', 0) == 0:
            missing.append('gsm8k')
        if quality.get('minerva_math', 0) == 0:
            missing.append('minerva_math')

        return missing

    def _get_active_processes(self) -> List[Dict[str, Any]]:
        """Identify currently active benchmarking processes"""
        import subprocess

        processes = []

        try:
            # Check for lm_eval processes
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=5
            )

            for line in result.stdout.split('\n'):
                if 'lm_eval' in line:
                    parts = line.split(None, 10)  # Get first 10 fields
                    if len(parts) >= 2:
                        processes.append({
                            'type': 'lm_eval',
                            'pid': parts[1],
                            'cpu': parts[2],
                            'command': parts[10] if len(parts) > 10 else ''
                        })
        except:
            pass

        return processes

    def generate_immediate_action_plan(self) -> Dict[str, Any]:
        """Generate concrete action plan for enterprise transformation"""

        models = self._get_all_model_states()

        complete_models = [m for m in models if m['complete']]
        incomplete_models = [m for m in models if not m['complete']]

        plan = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models': len(models),
                'complete': len(complete_models),
                'incomplete': len(incomplete_models),
                'ready_for_presentation': len(complete_models) >= 2
            },
            'immediate_actions': [],
            'phase_1_foundation': [],
            'phase_2_integration': [],
            'phase_3_enterprise': []
        }

        # PHASE 1: Foundation (Safe, no disruption)
        plan['phase_1_foundation'] = [
            "1. Create state management directory",
            "2. Save current system snapshot",
            "3. Add gemma-3-27b-awq to dashboard (Triton endpoint)",
            "4. Create backup of all existing cards",
            "5. Update dashboard with all available models"
        ]

        # PHASE 2: Integration (Careful, minimal interference)
        for model in incomplete_models:
            missing = model['missing_metrics']
            if 'perf' in missing:
                plan['phase_2_integration'].append(
                    f"6. Re-run perf tests for {model['name']} (safe retry)"
                )
            if 'humaneval' in missing and self._has_humaneval_results(model['name']):
                plan['phase_2_integration'].append(
                    f"7. Add humaneval to {model['name']} card (from lm_eval results)"
                )

        # PHASE 3: Enterprise (Complete functionality)
        plan['phase_3_enterprise'] = [
            "8. Create automatic health monitoring system",
            "9. Implement checkpoint/resume capability",
            "10. Add presentation-ready export formats",
            "11. Create executive summary dashboard",
            "12. Implement automatic recovery system"
        ]

        return plan

    def _has_humaneval_results(self, model_name: str) -> bool:
        """Check if humaneval lm_eval results exist"""
        # Map model names to their lm_eval directories
        lm_eval_dirs = {
            'qwen3-8b-bf16': 'out/lmeval_qwen3-8b-bf16_queue',
            'qwen2.5-vl-7b': 'out/lmeval_qwen2.5-vl-7b_queue',
            'glm-4-9b-chat': 'out/lmeval_glm_resumable',
            # Add more mappings as needed
        }

        if model_name in lm_eval_dirs:
            humaneval_dir = self.base_dir / lm_eval_dirs[model_name] / "humaneval"
            return humaneval_dir.exists()

        return False

    def print_action_plan(self):
        """Print formatted action plan"""
        plan = self.generate_immediate_action_plan()

        print("\n" + "="*80)
        print("🎯 ENTERPRISE TRANSFORMATION ACTION PLAN")
        print("="*80)

        print(f"\n📊 CURRENT STATUS:")
        print(f"   Total Models: {plan['summary']['total_models']}")
        print(f"   Complete: {plan['summary']['complete']}")
        print(f"   Incomplete: {plan['summary']['incomplete']}")
        print(f"   Ready for Presentation: {plan['summary']['ready_for_presentation']}")

        print(f"\n🚀 PHASE 1: FOUNDATION (Safe, No Disruption)")
        for i, action in enumerate(plan['phase_1_foundation'], 1):
            print(f"   {i}. {action}")

        if plan['phase_2_integration']:
            print(f"\n⚙️  PHASE 2: INTEGRATION (Careful, Minimal Interference)")
            for i, action in enumerate(plan['phase_2_integration'], len(plan['phase_1_foundation'])+1):
                print(f"   {i}. {action}")
        else:
            print(f"\n⚙️  PHASE 2: No immediate integration needed")

        print(f"\n🏢 PHASE 3: ENTERPRISE (Complete Functionality)")
        for i, action in enumerate(plan['phase_3_enterprise'], 1):
            print(f"   {i}. {action}")

        print("\n" + "="*80)

        return plan

def main():
    """Execute immediate safe steps"""
    plan = EnterprisePlan()

    print("🎯 Starting Enterprise Transformation...")

    # Step 1: Create directories
    plan.create_directories()

    # Step 2: Save snapshot
    plan.save_current_state_snapshot()

    # Step 3: Generate and print action plan
    action_plan = plan.print_action_plan()

    # Save action plan
    plan_file = plan.state_dir / "action_plan.json"
    plan_file.write_text(json.dumps(action_plan, indent=2, ensure_ascii=False))

    print(f"\n💾 Action plan saved: {plan_file}")
    print(f"📁 State directory: {plan.state_dir}")

    print("\n✅ READY FOR ENTERPRISE TRANSFORMATION")
    print("   No processes disrupted. All current work preserved.")
    print("   Can proceed safely with Phase 1 actions.")

if __name__ == "__main__":
    main()
