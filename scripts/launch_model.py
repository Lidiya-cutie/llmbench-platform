#!/usr/bin/env python3
"""
Единый скрипт запуска локальных LLM моделей через OpenAI-совместимые движки.

Читает models.yaml → использует engine_detect.py → запускает через Docker/LLAMA.CPP/Ollama.
Поддерживаемые движки: vLLM, SGLang, llama.cpp, HuggingFace TGI, Ollama, TensorRT-LLM.

Примеры:
  # Запуск модели из config
  python scripts/launch_model.py --model-key qwen3-30b-a3b --gpu-index 0

  # Остановка модели
  python scripts/launch_model.py --stop qwen3-30b-a3b

  # Список запущенных моделей
  python scripts/launch_model.py --list

  # Запуск с явным engine (игнорирует auto-detect)
  python scripts/launch_model.py --model-key local-27b-1bit --engine "llama.cpp"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmbench.engine_detect import detect_engine, validate_engine_config


@dataclass
class ProcessHandle:
    model_key: str
    pid: int
    port: int
    base_url: str
    served_model: str
    engine: str
    gpu_index: int
    launch_cmd: str
    log_file: str
    start_time: float

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "pid": self.pid,
            "port": self.port,
            "base_url": self.base_url,
            "served_model": self.served_model,
            "engine": self.engine,
            "gpu_index": self.gpu_index,
            "launch_cmd": self.launch_cmd,
            "log_file": self.log_file,
            "start_time": self.start_time,
            "uptime_s": time.monotonic() - self.start_time,
        }


LOGDIR = Path("/opt/llm-data/llmbench/logs")
PIDDIR = Path("/opt/llm-data/llmbench/logs/pids")


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_port_free(port: int) -> bool:
    try:
        result = subprocess.run(
            ["ss", "-tulpn"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTEN" in line:
                return False

        try:
            docker_result = subprocess.run(
                ["docker", "ps", "--format", "{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            for line in docker_result.stdout.splitlines():
                if f"0.0.0.0:{port}->" in line or f"::{port}->" in line:
                    return False
        except Exception:
            pass

        return True

    except Exception:
        return True


def free_port(port: int) -> None:
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
        time.sleep(1)

        docker_ps = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        for line in docker_ps.stdout.splitlines():
            if f"0.0.0.0:{port}->" in line or f"::{port}->" in line:
                container_name = line.split("\t")[0]
                if container_name:
                    print(f"  Остановка контейнера {container_name}...")
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        timeout=10,
                    )
                    time.sleep(2)

    except Exception:
        pass


def save_pid(handle: ProcessHandle) -> None:
    PIDDIR.mkdir(parents=True, exist_ok=True)
    pid_file = PIDDIR / f"{handle.model_key}.pid"
    with open(pid_file, "w") as f:
        json.dump(handle.to_dict(), f, indent=2)


def load_pid(model_key: str) -> Optional[ProcessHandle]:
    pid_file = PIDDIR / f"{model_key}.pid"
    if not pid_file.exists():
        return None

    try:
        with open(pid_file) as f:
            data = json.load(f)

        if not os.path.exists(f"/proc/{data['pid']}"):
            pid_file.unlink()
            return None

        return ProcessHandle(
            model_key=data["model_key"],
            pid=data["pid"],
            port=data["port"],
            base_url=data["base_url"],
            served_model=data["served_model"],
            engine=data["engine"],
            gpu_index=data["gpu_index"],
            launch_cmd=data["launch_cmd"],
            log_file=data["log_file"],
            start_time=data["start_time"],
        )
    except Exception:
        return None


def delete_pid(model_key: str) -> None:
    pid_file = PIDDIR / f"{model_key}.pid"
    if pid_file.exists():
        pid_file.unlink()


def launch_vllm(cmd: str, gpu_index: int, log_file: str) -> Optional[int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.pid


def launch_docker(cmd: str, log_file: str) -> Optional[int]:
    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.pid


def launch_llamacpp(cmd: str, log_file: str) -> Optional[int]:
    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.pid


def launch_ollama(cmd: str, gpu_index: int, log_file: str) -> Optional[int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["OLLAMA_MODELS"] = "/usr/share/ollama/.ollama/models"

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.pid


def launch_model(
    model_key: str,
    config_path: str = "config/models.yaml",
    gpu_index: int = 0,
    port: int = None,
    explicit_engine: str = None,
) -> Optional[ProcessHandle]:
    cfg = load_cfg(config_path)
    defaults = cfg.get("defaults", {})

    if model_key not in cfg["models"]:
        print(f"✗ Модель '{model_key}' не найдена в {config_path}")
        return None

    m = cfg["models"][model_key]

    if explicit_engine:
        m["engine"] = explicit_engine

    engine_cfg = detect_engine(m)
    warnings = validate_engine_config(engine_cfg)

    if warnings:
        for w in warnings:
            print(f"⚠ {w}")

    if not engine_cfg.launch_cmd:
        print(f"✗ Пустая launch_cmd для {model_key}")
        return None

    served = engine_cfg.served_model_name
    default_port = 8077

    if port is None:
        if "docker run" in engine_cfg.launch_cmd or "vllm serve" in engine_cfg.launch_cmd:
            default_port = 8077
        elif "llama-server" in engine_cfg.launch_cmd:
            default_port = 8080
        elif "ollama" in engine_cfg.launch_cmd:
            default_port = 11434
        elif "tgi" in engine_cfg.launch_cmd.lower():
            default_port = 8079
        elif "sglang" in engine_cfg.launch_cmd.lower():
            default_port = 8078

    port = port or default_port

    if not check_port_free(port):
        print(f"✗ Порт {port} занят. Используйте --port или остановите процесс.")
        return None

    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGDIR / f"{model_key}_{int(time.time())}.log"

    cmd = engine_cfg.launch_cmd

    if "vllm serve" in cmd and "docker" not in cmd:
        pid = launch_vllm(cmd, gpu_index, str(log_file))
    elif "docker run" in cmd:
        cmd = cmd.replace(f"-p {port}:", f"-p {port}:{port}:")
        pid = launch_docker(cmd, str(log_file))
    elif "llama-server" in cmd or "./build/bin/" in cmd:
        pid = launch_llamacpp(cmd, str(log_file))
    elif "ollama" in cmd:
        pid = launch_ollama(cmd, gpu_index, str(log_file))
    else:
        pid = launch_vllm(cmd, gpu_index, str(log_file))

    if pid is None:
        print(f"✗ Не удалось запустить процесс для {model_key}")
        return None

    handle = ProcessHandle(
        model_key=model_key,
        pid=pid,
        port=port,
        base_url=f"http://127.0.0.1:{port}/v1",
        served_model=served,
        engine=engine_cfg.engine,
        gpu_index=gpu_index,
        launch_cmd=cmd,
        log_file=str(log_file),
        start_time=time.monotonic(),
    )

    save_pid(handle)

    print(f"✓ Запущен {model_key}")
    print(f"  PID: {pid}")
    print(f"  Port: {port}")
    print(f"  URL: {handle.base_url}")
    print(f"  Model: {served}")
    print(f"  Engine: {engine_cfg.engine}")
    print(f"  Log: {log_file}")

    return handle


def stop_model(model_key: str) -> bool:
    handle = load_pid(model_key)

    if handle is None:
        print(f"✗ Модель '{model_key}' не найдена в запущенных процессах")
        return False

    try:
        os.kill(handle.pid, 15)
        time.sleep(2)

        if os.path.exists(f"/proc/{handle.pid}"):
            os.kill(handle.pid, 9)
            time.sleep(1)

        delete_pid(model_key)
        print(f"✓ Остановлен {model_key} (PID {handle.pid})")
        return True

    except Exception as e:
        print(f"✗ Ошибка остановки {model_key}: {e}")
        return False


def list_running() -> List[ProcessHandle]:
    running = []

    for pid_file in PIDDIR.glob("*.pid"):
        model_key = pid_file.stem
        handle = load_pid(model_key)

        if handle is not None:
            running.append(handle)

    return running


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--model-key", help="ключ модели из models.yaml")
    ap.add_argument("--engine", help="явный engine (игнорирует auto-detect)")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--port", type=int)
    ap.add_argument("--stop", metavar="MODEL_KEY", help="остановить модель")
    ap.add_argument("--list", action="store_true", help="список запущенных моделей")
    args = ap.parse_args()

    if args.list:
        running = list_running()

        if not running:
            print("Нет запущенных моделей")
            return

        print(f"Запущено моделей: {len(running)}\n")

        for h in running:
            uptime = int(time.monotonic() - h.start_time)
            print(f"{h.model_key}:")
            print(f"  PID: {h.pid}")
            print(f"  URL: {h.base_url}")
            print(f"  Model: {h.served_model}")
            print(f"  Engine: {h.engine}")
            print(f"  Uptime: {uptime // 60}m {uptime % 60}s")
            print(f"  Log: {h.log_file}")
            print()

        return

    if args.stop:
        stop_model(args.stop)
        return

    if not args.model_key:
        ap.print_help()
        return

    launch_model(
        model_key=args.model_key,
        config_path=args.config,
        gpu_index=args.gpu_index,
        port=args.port,
        explicit_engine=args.engine,
    )


if __name__ == "__main__":
    main()
