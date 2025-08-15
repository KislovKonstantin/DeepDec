from deepdec import load_config, SystemOrchestrator
import json

def test_full_e2e_flow():
    config = load_config("config.json")
    orchestrator = (
        SystemOrchestrator.Builder(config)
        .with_workflow("analysis", "analysis", "eval_analysis")
        .with_workflow("commentary", "commentary", "eval_comment")
        .with_workflow("reconstruction", "reconstruct", "eval_reconstruct")
        .with_workflow("aggregator", "aggregator", "eval_aggregator")
        .build()
    )
    input_file = "test.txt"
    output_file = "full.json"
    orchestrator.execute(input_file, str(output_file))
    with open(output_file) as f:
        result = json.load(f)
        assert "aggregated" in result and (result['reconstruction']=='No valid output' or '}' in result['reconstruction'])
