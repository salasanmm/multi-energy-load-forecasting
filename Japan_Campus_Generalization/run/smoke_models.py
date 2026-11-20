import torch

import run_japan_811 as runner


for name in runner.MODELS:
    model, _ = runner.config_for(name, 24)
    model.eval()
    sample = torch.randn(2, 168, len(runner.INPUT_FEATURES))
    input_adapter = torch.nn.Linear(len(runner.INPUT_FEATURES), len(runner.TARGET_COLUMNS))
    with torch.no_grad():
        input_adapter.weight.zero_()
        input_adapter.bias.zero_()
        input_adapter.weight[:, :len(runner.TARGET_COLUMNS)] = torch.eye(len(runner.TARGET_COLUMNS))
    with torch.no_grad():
        output = runner.forward(name, model, sample, 24, input_adapter)
    params = (sum(p.numel() for p in model.parameters() if p.requires_grad)
              + sum(p.numel() for p in input_adapter.parameters() if p.requires_grad))
    if output.shape != (2, 24, 3):
        raise RuntimeError(f"{name} returned {tuple(output.shape)}, expected (2, 24, 3)")
    print(f"{name}: output={tuple(output.shape)}, params={params}")
