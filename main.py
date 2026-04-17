import yaml
from Orchestrator import ModelOrchestrator
from checker import BackgroundChecker
from shell import Shell

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    orch = ModelOrchestrator(config=config)
    interval = config.get("system", {}).get("monitor_interval", 300)
    bg = BackgroundChecker(orch, monitor_interval=interval)
    com = Shell(bg)
    com.cmdloop()