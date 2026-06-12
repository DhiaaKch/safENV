"""
Simulateur de logs Windows en pseudo temps reel pour le projet UEBA.

Objectif:
- Generer des evenements bruts proches de vraies sources entreprise.
- Couvrir les familles de signaux utiles pour model_3.py:
  logon_*, device_*, file_*, http_*, email_*, escalation_score, exfiltration_score
- Sortir du JSON Lines exploitable par Filebeat / Logstash / Elasticsearch.

Exemples:
  python scripts/simulate_windows_realtime_logs.py
  python scripts/simulate_windows_realtime_logs.py --users 25 --sleep 0.25
  python scripts/simulate_windows_realtime_logs.py --duration-minutes 5 --output data/sim/windows_events.jsonl
  python scripts/simulate_windows_realtime_logs.py --es-url http://localhost:9200 --es-index raw-events-windows-sim
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


RANDOM_SEED = 42
DOMAIN = "corp.local"
DEPARTMENTS = ["Finance", "HR", "IT", "Legal", "Sales", "Operations"]
HOST_PREFIXES = ["WIN-CL", "WIN-LT", "SRV-JMP"]
INTERNAL_DOMAINS = ["intranet.corp.local", "sharepoint.corp.local", "hr.corp.local"]
SENSITIVE_DOMAINS = ["wikileaks.org", "mega.nz", "dropbox.com", "we-transfer.com"]
JOB_DOMAINS = ["linkedin.com", "indeed.com", "glassdoor.com"]
FILE_SHARING_DOMAINS = ["drive.google.com", "dropbox.com", "mega.nz", "we-transfer.com"]
FILE_EXTENSIONS = [".docx", ".xlsx", ".pdf", ".zip", ".pptx", ".csv"]
USB_LABELS = ["Kingston_DT", "SanDisk_Ultra", "Generic_USB", "SecureKey"]
PROCESS_BASELINE = [
    ("outlook.exe", "explorer.exe"),
    ("teams.exe", "explorer.exe"),
    ("chrome.exe", "explorer.exe"),
    ("excel.exe", "explorer.exe"),
    ("winword.exe", "explorer.exe"),
]
PROCESS_SUSPICIOUS = [
    ("powershell.exe", "winword.exe"),
    ("7z.exe", "powershell.exe"),
    ("robocopy.exe", "cmd.exe"),
    ("certutil.exe", "powershell.exe"),
]
DEPARTMENT_TITLES = {
    "IT": ["IT Admin", "System Engineer", "Security Analyst", "Infrastructure Admin"],
    "Finance": ["Finance Analyst", "Controller", "Accounting Manager"],
    "HR": ["HR Specialist", "Recruiter", "HR Manager"],
    "Legal": ["Legal Counsel", "Paralegal", "Compliance Officer"],
    "Sales": ["Sales Representative", "Account Executive", "Sales Manager"],
    "Operations": ["Operations Analyst", "Operations Manager", "Coordinator"],
}


@dataclass
class UserProfile:
    user_id: str
    full_name: str
    department: str
    title: str
    host: str
    insider: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation temps reel de logs Windows UEBA.")
    parser.add_argument("--users", type=int, default=20, help="Nombre d'utilisateurs simules.")
    parser.add_argument(
        "--insider-ratio",
        type=float,
        default=0.10,
        help="Proportion d'utilisateurs avec scenario suspect.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Temps d'attente reel entre batches d'evenements.",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=5,
        help="Avancement de l'horloge simulee a chaque batch.",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=60,
        help="Duree totale simulee; 0 = boucle infinie.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Debut simulation ISO-8601 UTC. Defaut: maintenant UTC.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sim/windows_realtime_events.jsonl",
        help="Fichier JSONL de sortie.",
    )
    parser.add_argument(
        "--es-url",
        type=str,
        default=None,
        help="URL Elasticsearch, ex: http://localhost:9200",
    )
    parser.add_argument(
        "--es-index",
        type=str,
        default="raw-events-windows-sim",
        help="Index Elasticsearch cible si --es-url est fourni.",
    )
    parser.add_argument(
        "--send-logstash",
        action="store_true",
        help="Envoyer chaque batch vers Logstash en JSON lines TCP.",
    )
    parser.add_argument(
        "--logstash-host",
        type=str,
        default="localhost",
        help="Hote Logstash TCP.",
    )
    parser.add_argument(
        "--logstash-port",
        type=int,
        default=5000,
        help="Port Logstash TCP json_lines.",
    )
    return parser.parse_args()


def build_users(count: int, insider_ratio: float, rng: random.Random) -> List[UserProfile]:
    users: List[UserProfile] = []
    insider_count = max(1, int(round(count * insider_ratio))) if count > 0 else 0
    for idx in range(count):
        uid = f"user{idx + 1:03d}"
        department = rng.choice(DEPARTMENTS)
        title = rng.choice(DEPARTMENT_TITLES[department])
        users.append(
            UserProfile(
                user_id=uid,
                full_name=f"User {idx + 1:03d}",
                department=department,
                title=title,
                host=f"{rng.choice(HOST_PREFIXES)}-{idx + 1:03d}",
                insider=idx < insider_count,
            )
        )
    rng.shuffle(users)
    return users


def parse_start(raw: Optional[str]) -> datetime:
    if not raw:
        return datetime.now(timezone.utc).replace(second=0, microsecond=0)
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def base_event(ts: datetime, profile: UserProfile, source: str, category: str, action: str) -> Dict:
    return {
        "@timestamp": ts.isoformat().replace("+00:00", "Z"),
        "host": {"name": profile.host, "os": {"family": "windows"}},
        "user": {
            "name": profile.user_id,
            "full_name": profile.full_name,
            "role": profile.title,
            "roles": [profile.title],
        },
        "role": profile.title,
        "title": profile.title,
        "organization": {"department": profile.department},
        "source": {"system": source},
        "event": {
            "category": category,
            "action": action,
            "kind": "event",
            "outcome": "success",
        },
        "labels": {
            "simulation": True,
            "profile_type": "insider" if profile.insider else "normal",
        },
    }


def logon_event(ts: datetime, profile: UserProfile, afterhours: bool) -> Dict:
    event_id = 4624 if rng_bool(0.96) else 4625
    event = base_event(ts, profile, "windows_security", "authentication", "logon")
    event["winlog"] = {"channel": "Security", "event_id": event_id}
    event["event"]["outcome"] = "success" if event_id == 4624 else "failure"
    event["logon"] = {
        "type": "interactive",
        "is_afterhours": afterhours,
        "source_ip": f"10.20.{random.randint(1, 20)}.{random.randint(10, 240)}",
    }
    return event


def process_event(ts: datetime, profile: UserProfile, suspicious: bool) -> Dict:
    process_name, parent_name = random.choice(PROCESS_SUSPICIOUS if suspicious else PROCESS_BASELINE)
    event = base_event(ts, profile, "sysmon", "process", "start")
    event["winlog"] = {"channel": "Microsoft-Windows-Sysmon/Operational", "event_id": 1}
    event["process"] = {
        "name": process_name,
        "parent": {"name": parent_name},
        "command_line": build_command_line(process_name, suspicious),
    }
    if suspicious:
        event["ueba"] = {"escalation_hint": 1}
    return event


def web_event(ts: datetime, profile: UserProfile, suspicious: bool) -> Dict:
    if suspicious:
        bucket = random.choices(
            ["sensitive", "file_sharing", "job_search"],
            weights=[0.50, 0.25, 0.25],
            k=1,
        )[0]
    else:
        bucket = random.choices(
            ["internal", "external", "job_search"],
            weights=[0.65, 0.30, 0.05],
            k=1,
        )[0]

    if bucket == "sensitive":
        domain = random.choice(SENSITIVE_DOMAINS)
    elif bucket == "file_sharing":
        domain = random.choice(FILE_SHARING_DOMAINS)
    elif bucket == "job_search":
        domain = random.choice(JOB_DOMAINS)
    elif bucket == "internal":
        domain = random.choice(INTERNAL_DOMAINS)
    else:
        domain = random.choice(["microsoft.com", "github.com", "news.ycombinator.com"])

    event = base_event(ts, profile, "proxy", "network", "http_request")
    event["url"] = {
        "full": f"https://{domain}/{random.choice(['home', 'download', 'jobs', 'docs', 'upload'])}",
        "domain": domain,
    }
    event["http"] = {
        "request": {"method": random.choice(["GET", "POST"])},
        "response": {"status_code": random.choice([200, 200, 200, 302, 404])},
    }
    event["network"] = {"protocol": "http", "bytes_out": random.randint(800, 75000)}
    event["ueba"] = {
        "http_wikileaks_flag": int(domain == "wikileaks.org"),
        "http_job_search_flag": int(domain in JOB_DOMAINS),
        "http_file_sharing_flag": int(domain in FILE_SHARING_DOMAINS),
    }
    return event


def file_event(ts: datetime, profile: UserProfile, suspicious: bool) -> Dict:
    action = random.choices(["read", "copy", "write"], weights=[0.50, 0.35, 0.15], k=1)[0]
    if suspicious and random.random() < 0.55:
        action = "copy"
    extension = random.choice(FILE_EXTENSIONS)
    folder = random.choice(["Finance", "HR", "Projects", "Legal", "Shared"])
    event = base_event(ts, profile, "sysmon", "file", action)
    event["winlog"] = {"channel": "Microsoft-Windows-Sysmon/Operational", "event_id": 11}
    event["file"] = {
        "path": f"C:\\Users\\{profile.user_id}\\Documents\\{folder}\\file_{random.randint(100,999)}{extension}",
        "extension": extension,
        "size": random.randint(20_000, 8_000_000),
    }
    event["ueba"] = {"file_copy_flag": int(action == "copy")}
    return event


def device_event(ts: datetime, profile: UserProfile, suspicious: bool) -> Dict:
    event = base_event(ts, profile, "windows_pnp", "device", "connect")
    event["winlog"] = {"channel": "Microsoft-Windows-DriverFrameworks-UserMode/Operational", "event_id": 2003}
    event["device"] = {
        "type": "usb_storage",
        "label": random.choice(USB_LABELS),
        "serial": f"USB-{random.randint(100000, 999999)}",
    }
    event["ueba"] = {"device_risk_hint": 2 if suspicious else 1}
    return event


def email_event(ts: datetime, profile: UserProfile, suspicious: bool) -> Dict:
    external = suspicious or random.random() < 0.25
    recipient_domain = random.choice(
        ["gmail.com", "outlook.com", "proton.me"] if external else [DOMAIN]
    )
    event = base_event(ts, profile, "m365", "email", "send")
    event["email"] = {
        "subject": random.choice(
            [
                "Meeting notes",
                "Quarterly update",
                "Draft contract",
                "Shared documents",
                "Please review",
            ]
        ),
        "to": [f"recipient{random.randint(1, 50)}@{recipient_domain}"],
        "attachment_count": random.choices([0, 1, 2, 3], weights=[0.40, 0.35, 0.20, 0.05], k=1)[0],
        "is_external": external,
    }
    event["ueba"] = {"email_external_flag": int(external)}
    return event


def privilege_event(ts: datetime, profile: UserProfile) -> Dict:
    event = base_event(ts, profile, "windows_security", "iam", "privilege_assigned")
    event["winlog"] = {"channel": "Security", "event_id": random.choice([4672, 4728, 4732])}
    event["privilege"] = {"name": random.choice(["SeDebugPrivilege", "Backup Operators", "Administrators"])}
    event["ueba"] = {"escalation_hint": 2}
    return event


def build_command_line(process_name: str, suspicious: bool) -> str:
    if process_name == "powershell.exe":
        if suspicious:
            return "powershell.exe -ExecutionPolicy Bypass -File collect_and_zip.ps1"
        return "powershell.exe -File login_script.ps1"
    if process_name == "7z.exe":
        return "7z.exe a archive.zip C:\\Users\\Public\\Exports\\*"
    if process_name == "robocopy.exe":
        return "robocopy.exe C:\\Sensitive E:\\Backup /E"
    if process_name == "certutil.exe":
        return "certutil.exe -encode input.zip output.b64"
    return f"{process_name} /background"


def rng_bool(probability: float) -> bool:
    return random.random() < probability


def iter_events(users: List[UserProfile], start_ts: datetime, step_minutes: int) -> Iterator[List[Dict]]:
    current = start_ts
    while True:
        batch: List[Dict] = []
        for profile in users:
            hour = current.hour
            business_hours = 8 <= hour <= 18
            afterhours = not business_hours

            if business_hours:
                base_prob = 0.75
            else:
                base_prob = 0.12 if not profile.insider else 0.30

            if random.random() > base_prob:
                continue

            batch.append(logon_event(current, profile, afterhours))

            process_count = random.randint(1, 3 if business_hours else 2)
            for _ in range(process_count):
                batch.append(process_event(jitter_ts(current, 90), profile, suspicious=profile.insider and afterhours))

            web_count = random.randint(1, 4 if business_hours else 2)
            if profile.insider and (afterhours or random.random() < 0.35):
                web_count += random.randint(2, 6)
            for _ in range(web_count):
                batch.append(web_event(jitter_ts(current, 140), profile, suspicious=profile.insider))

            if random.random() < (0.18 if not profile.insider else 0.45):
                batch.append(file_event(jitter_ts(current, 180), profile, suspicious=profile.insider))

            if random.random() < (0.03 if not profile.insider else 0.22):
                batch.append(device_event(jitter_ts(current, 220), profile, suspicious=profile.insider))

            if random.random() < (0.08 if not profile.insider else 0.28):
                mail_events = 1 if not profile.insider else random.randint(1, 4)
                for _ in range(mail_events):
                    batch.append(email_event(jitter_ts(current, 260), profile, suspicious=profile.insider))

            if profile.insider and random.random() < 0.10:
                batch.append(privilege_event(jitter_ts(current, 120), profile))

        yield sorted(batch, key=lambda item: item["@timestamp"])
        current = current + timedelta(minutes=step_minutes)


def jitter_ts(base: datetime, max_seconds: int) -> datetime:
    return base + timedelta(seconds=random.randint(1, max_seconds))


def write_jsonl(path: Path, events: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def post_bulk_events(es_url: str, index: str, events: List[Dict], timeout: int = 8) -> None:
    if not events:
        return
    lines: List[str] = []
    for event in events:
        lines.append(json.dumps({"index": {"_index": index}}, ensure_ascii=True))
        lines.append(json.dumps(event, ensure_ascii=True))
    body = ("\n".join(lines) + "\n").encode("utf-8")

    req = urllib.request.Request(
        url=f"{es_url.rstrip('/')}/_bulk",
        data=body,
        headers={"Content-Type": "application/x-ndjson"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if payload.get("errors"):
                print("WARN Elasticsearch bulk returned errors", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"WARN impossible d'envoyer vers Elasticsearch: {exc}", file=sys.stderr)


def send_events_logstash(host: str, port: int, events: List[Dict], timeout: int = 5) -> None:
    if not events:
        return
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            for event in events:
                line = json.dumps(event, ensure_ascii=True) + "\n"
                sock.sendall(line.encode("utf-8"))
    except OSError as exc:
        print(f"WARN impossible d'envoyer vers Logstash {host}:{port}: {exc}", file=sys.stderr)


def summarize_batch(events: List[Dict]) -> str:
    counts = {
        "auth": 0,
        "process": 0,
        "http": 0,
        "file": 0,
        "device": 0,
        "email": 0,
        "iam": 0,
    }
    for event in events:
        category = event["event"]["category"]
        counts[category] = counts.get(category, 0) + 1
    return " ".join(f"{key}={value}" for key, value in counts.items() if value)


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)

    users = build_users(args.users, args.insider_ratio, random)
    start_ts = parse_start(args.start)
    output_path = Path(args.output)

    print(
        f"Simulation Windows UEBA start={start_ts.isoformat()} users={len(users)} "
        f"sleep={args.sleep}s step={args.step_minutes}min output={output_path}"
    )
    print(
        f"Utilisateurs insiders: {sum(1 for user in users if user.insider)} / {len(users)}"
    )

    generator = iter_events(users, start_ts, args.step_minutes)
    remaining = args.duration_minutes

    while True:
        if remaining == 0 and args.duration_minutes > 0:
            break

        batch = next(generator)
        if batch:
            write_jsonl(output_path, batch)
            if args.es_url:
                post_bulk_events(args.es_url, args.es_index, batch)
            if args.send_logstash:
                send_events_logstash(args.logstash_host, args.logstash_port, batch)
            print(
                f"[{batch[0]['@timestamp']} .. {batch[-1]['@timestamp']}] "
                f"events={len(batch)} {summarize_batch(batch)}"
            )
        else:
            print("batch vide")

        if args.duration_minutes > 0:
            remaining = max(0, remaining - args.step_minutes)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
