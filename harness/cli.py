import argparse


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Harness — local agent loop")
    parser.add_argument("prompt", nargs="?", help="Message to send to the model")
    parser.add_argument("--session", default=None, help="Session ID (default: from config)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--chat", action="store_true", help="Interactive multi-turn chat mode")
    return parser.parse_args(argv)
