#!/usr/bin/env python3
"""Build the NL -> shell command dataset in mlx-lm chat format.

Produces train.jsonl / valid.jsonl / test.jsonl next to this file.

The data is a hand-curated seed set of (instruction, command) pairs, lightly
augmented with natural-language paraphrases of each instruction. Curated
quality + a clean train/valid/test split is the whole point: it makes the
eval numbers trustworthy.

Run:  python data/build_dataset.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "You are a shell command generator for macOS/Linux. Given a task in plain "
    "English, reply with a single safe shell command that accomplishes it. "
    "Output only the command on one line, with no explanation and no markdown."
)

# ---------------------------------------------------------------------------
# Curated seed pairs: (instruction, command)
# Kept deliberately practical and unambiguous so a reference command is fair.
# ---------------------------------------------------------------------------
SEEDS = [
    # --- listing / navigation ---
    ("list all files in the current directory including hidden ones", "ls -la"),
    ("list files sorted by modification time, newest first", "ls -lt"),
    ("list files sorted by size, largest first", "ls -lS"),
    ("show the current working directory", "pwd"),
    ("show the directory tree two levels deep", "find . -maxdepth 2 -print"),
    ("count the number of files in the current directory", "ls -1 | wc -l"),
    ("list only directories in the current folder", "ls -d */"),
    ("show the 10 largest files under the current directory", "find . -type f -exec du -h {} + | sort -rh | head -n 10"),

    # --- find ---
    ("find all python files under the current directory", "find . -name '*.py'"),
    ("find all files larger than 100 megabytes", "find . -type f -size +100M"),
    ("find files modified in the last 24 hours", "find . -type f -mtime -1"),
    ("find empty files in the current directory tree", "find . -type f -empty"),
    ("find and delete all .DS_Store files", "find . -name '.DS_Store' -delete"),
    ("find all files owned by root under /tmp", "find /tmp -user root"),
    ("find files with permission 777", "find . -type f -perm 777"),

    # --- grep / text search ---
    ("search for the word TODO in all files recursively", "grep -rn 'TODO' ."),
    ("search for 'error' case-insensitively in app.log", "grep -i 'error' app.log"),
    ("count how many lines contain the word warning in app.log", "grep -c 'warning' app.log"),
    ("show lines that do not contain the word debug in app.log", "grep -v 'debug' app.log"),
    ("find which files contain the string api_key", "grep -rl 'api_key' ."),
    ("show 3 lines of context around matches of 'panic' in server.log", "grep -C 3 'panic' server.log"),

    # --- text processing ---
    ("show the first 20 lines of access.log", "head -n 20 access.log"),
    ("show the last 50 lines of access.log", "tail -n 50 access.log"),
    ("follow access.log in real time", "tail -f access.log"),
    ("print the third column of a comma separated file data.csv", "cut -d',' -f3 data.csv"),
    ("count the number of words in notes.txt", "wc -w notes.txt"),
    ("sort the lines of names.txt alphabetically", "sort names.txt"),
    ("show unique lines in items.txt", "sort items.txt | uniq"),
    ("count occurrences of each unique line in items.txt", "sort items.txt | uniq -c"),
    ("replace all tabs with commas in data.txt and print to stdout", "tr '\\t' ',' < data.txt"),
    ("print lines 10 through 20 of file.txt", "sed -n '10,20p' file.txt"),
    ("replace foo with bar in config.txt in place", "sed -i '' 's/foo/bar/g' config.txt"),

    # --- file ops ---
    ("create an empty file called notes.txt", "touch notes.txt"),
    ("create a directory called build", "mkdir build"),
    ("create nested directories src/utils/helpers", "mkdir -p src/utils/helpers"),
    ("copy report.pdf to the backups folder", "cp report.pdf backups/"),
    ("copy the src directory recursively to dist", "cp -r src dist"),
    ("move draft.txt to final.txt", "mv draft.txt final.txt"),
    ("delete the file temp.log", "rm temp.log"),
    ("remove the empty directory called old", "rmdir old"),
    ("rename all .txt files to .md in the current directory", "for f in *.txt; do mv \"$f\" \"${f%.txt}.md\"; done"),
    ("make the script deploy.sh executable", "chmod +x deploy.sh"),
    ("create a symbolic link named latest pointing to v2", "ln -s v2 latest"),

    # --- archives ---
    ("create a gzip tar archive of the src folder named src.tar.gz", "tar -czf src.tar.gz src"),
    ("extract the archive backup.tar.gz", "tar -xzf backup.tar.gz"),
    ("list the contents of archive.tar.gz without extracting", "tar -tzf archive.tar.gz"),
    ("zip the reports folder into reports.zip", "zip -r reports.zip reports"),
    ("unzip data.zip into the data directory", "unzip data.zip -d data"),

    # --- disk / system ---
    ("show disk usage of the current directory in human readable form", "du -sh ."),
    ("show free disk space in human readable form", "df -h"),
    ("show the size of each subdirectory in the current folder", "du -sh */"),
    ("show currently running processes", "ps aux"),
    ("find the process listening on port 8080", "lsof -i :8080"),
    ("kill the process with pid 1234", "kill 1234"),
    ("force kill all processes named node", "pkill -9 node"),
    ("show memory usage", "top -l 1 | head -n 10"),
    ("show the current user", "whoami"),
    ("show how long the system has been up", "uptime"),

    # --- networking ---
    ("download the file at https://example.com/data.zip", "curl -O https://example.com/data.zip"),
    ("make a get request to https://api.example.com/health", "curl https://api.example.com/health"),
    ("check if google.com is reachable with 4 pings", "ping -c 4 google.com"),
    ("show my public ip address", "curl ifconfig.me"),
    ("send a json post request to https://api.example.com/users", "curl -X POST -H 'Content-Type: application/json' -d '{}' https://api.example.com/users"),

    # --- git ---
    ("show the git status", "git status"),
    ("stage all changes", "git add -A"),
    ("commit with the message initial commit", "git commit -m 'initial commit'"),
    ("show the last 5 commits in one line each", "git log --oneline -5"),
    ("create and switch to a new branch called feature", "git checkout -b feature"),
    ("discard all uncommitted changes in the working tree", "git checkout -- ."),
    ("show what changed in the last commit", "git show HEAD"),
    ("pull the latest changes from origin main", "git pull origin main"),

    # --- permissions / ownership ---
    ("give the owner read write execute and others read on file.sh", "chmod 744 file.sh"),
    ("recursively change ownership of the app folder to user deploy", "chown -R deploy app"),
    ("make all .sh files in the current directory executable", "chmod +x *.sh"),

    # --- env / misc ---
    ("print the value of the PATH environment variable", "echo $PATH"),
    ("set an environment variable named API_KEY to abc123 for this session", "export API_KEY=abc123"),
    ("show the current date in YYYY-MM-DD format", "date +%Y-%m-%d"),
    ("show the calendar for this month", "cal"),
    ("count the number of cpu cores", "sysctl -n hw.ncpu"),
    ("print hello world", "echo 'hello world'"),
    ("clear the terminal screen", "clear"),
    ("show the manual page for the tar command", "man tar"),
    ("show command history", "history"),
    ("repeat the ls command every 2 seconds", "watch -n 2 ls"),
]

# Natural-language paraphrase prefixes used to augment instructions so the
# model learns the task, not one phrasing. Applied probabilistically.
PARAPHRASE_PREFIXES = [
    "{i}",
    "{i}",                       # weight the identity form
    "how do I {i}",
    "can you {i}",
    "I want to {i}",
    "please {i}",
    "command to {i}",
    "show me how to {i}",
]


def to_record(instruction: str, command: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": command},
        ]
    }


def main() -> None:
    rng = random.Random(1707)  # deterministic build

    records = []
    seen = set()
    for instruction, command in SEEDS:
        # always include the canonical phrasing
        forms = {instruction}
        # add 1-2 paraphrases per seed
        for tmpl in rng.sample(PARAPHRASE_PREFIXES, k=3):
            forms.add(tmpl.format(i=instruction))
        for form in forms:
            key = (form, command)
            if key in seen:
                continue
            seen.add(key)
            records.append(to_record(form, command))

    rng.shuffle(records)

    n = len(records)
    n_test = max(12, int(n * 0.12))
    n_valid = max(8, int(n * 0.10))
    test = records[:n_test]
    valid = records[n_test:n_test + n_valid]
    train = records[n_test + n_valid:]

    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        path = HERE / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows):4d} -> {path.name}")

    print(f"\ntotal {n} records from {len(SEEDS)} curated seeds")


if __name__ == "__main__":
    main()
