#!/usr/bin/env python3
"""Non-interactive front end for the epiq issue tracker (https://github.com/ljtn/epiq).

epiq is a TUI; its scriptable surface is the `epiq-mcp` stdio server. This script
speaks just enough JSON-RPC to drive it from a shell, so that every step of the
analysis can be opened, commented on and closed as a ticket.

    epiq.py new "<title>" [--goal G] [--tags a,b] [--desc TEXT] [--lane todo|doing|done]
    epiq.py start <ref>                 # -> In progress
    epiq.py comment <ref> "<markdown>"
    epiq.py done <ref> ["<comment>"]    # -> Done
    epiq.py list
    epiq.py export <dir>                # board.md + events.json snapshot
    epiq.py call <tool> '<json>'        # raw MCP tool call

Commits link to a ticket when their subject starts with the ticket ref.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

ACTOR = os.environ.get("EPIQ_ACTOR", "claude/ada")
LANES = {"todo": "Todo", "doing": "In progress", "done": "Done"}


class Epiq:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["epiq-mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        self.n = 0
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "epiq.py", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.call("epiq_actor_assume", name=ACTOR)

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method, params):
        self.n += 1
        self._send({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                sys.exit("epiq-mcp closed the connection")
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if reply.get("id") == self.n:
                return reply

    def call(self, tool, **args):
        res = self._rpc("tools/call", {"name": tool, "arguments": args}).get("result", {})
        text = "".join(c.get("text", "") for c in res.get("content", []))
        if res.get("isError"):
            sys.exit(f"{tool} failed: {text}")
        try:
            return json.loads(text).get("value")
        except ValueError:
            return text

    def lane_id(self, key):
        for lane in self.call("epiq_swimlane_list"):
            if lane["title"] == LANES[key]:
                return lane["id"]
        sys.exit(f"no lane {key}")

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()


def export(ep, out_dir):
    """Write a human-readable board snapshot and the raw event log."""
    os.makedirs(out_dir, exist_ok=True)
    state = ep.call("epiq_state_get")
    nodes = state["nodes"]
    users = {c["id"]: c["name"] for c in ep.call("epiq_contributor_list")}
    lanes = {l["id"]: l["title"] for l in ep.call("epiq_swimlane_list")}
    issues = ep.call("epiq_issue_list", brief=False, includeClosed=True)
    goals = ep.call("epiq_goal_list")

    def ts(ms):
        return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

    md = ["# epiq board snapshot", "",
          "Exported from the [epiq](https://github.com/ljtn/epiq) event log with "
          "`tools/epiq.py export`. Tickets are listed in creation order; commits whose "
          "subject starts with a ticket ref belong to that ticket "
          "(`git log --oneline --grep '^<ref>'`).", ""]
    if goals:
        md += ["## Goals", "", "| Goal | Done / total |", "|---|---|"]
        md += [f"| {g.get('name', g.get('title'))} | {g.get('done')} / {g.get('total')} |" for g in goals]
        md.append("")
    md += ["## Tickets", "", "| Ref | Lane | Goal | Tags | Title |", "|---|---|---|---|---|"]
    issues.sort(key=lambda i: i["createdAt"])
    for i in issues:
        goal = (i.get("goal") or {}).get("name", "") if isinstance(i.get("goal"), dict) else (i.get("goal") or "")
        tags = ", ".join(t["name"] for t in i.get("tags", []))
        md.append(f"| `{i['id']}` | {lanes.get(i['parentNodeId'], '?')} | {goal} | {tags} | {i['title']} |")
    md.append("")
    for i in issues:
        full = ep.call("epiq_issue_get", idOrRef=i["id"], allComments=True)
        md += [f"### `{i['id']}` {i['title']}", "",
               f"*Opened {ts(i['createdAt'])} UTC — lane: {lanes.get(i['parentNodeId'], '?')}*", ""]
        if i.get("description"):
            md += [i["description"], ""]
        for c in full.get("comments", []):
            who = users.get(c.get("userId"), c.get("author", c.get("userName", "")))
            body = c.get("body", c.get("text", ""))
            md += [f"> **{who}**: " + body.replace("\n", "\n> "), ""]
    with open(os.path.join(out_dir, "board.md"), "w") as fh:
        fh.write("\n".join(md))
    events = sorted((e for n in nodes.values() for e in n.get("log", [])), key=lambda e: e["id"])
    seen, uniq = set(), []
    for e in events:
        if e["id"] not in seen:
            seen.add(e["id"])
            uniq.append(e)
    with open(os.path.join(out_dir, "events.json"), "w") as fh:
        json.dump(uniq, fh, indent=1)
    print(f"exported {len(issues)} tickets, {len(uniq)} events -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("title"); p.add_argument("--goal")
    p.add_argument("--tags", default=""); p.add_argument("--desc", default=""); p.add_argument("--lane", default="todo")
    p = sub.add_parser("start"); p.add_argument("ref")
    p = sub.add_parser("comment"); p.add_argument("ref"); p.add_argument("body")
    p = sub.add_parser("done"); p.add_argument("ref"); p.add_argument("body", nargs="?")
    sub.add_parser("list")
    p = sub.add_parser("export"); p.add_argument("dir")
    p = sub.add_parser("call"); p.add_argument("tool"); p.add_argument("json", nargs="?", default="{}")
    a = ap.parse_args()

    ep = Epiq()
    if a.cmd == "new":
        kw = dict(title=a.title, parentId=ep.lane_id(a.lane))
        if a.desc:
            kw["description"] = a.desc
        if a.tags:
            kw["tagNames"] = a.tags.split(",")
        issue = ep.call("epiq_issue_create", **kw)
        if a.goal:
            ep.call("epiq_issue_goal_set", issueId=issue["id"], goal=a.goal)
        print(issue["id"])
    elif a.cmd == "start":
        ep.call("epiq_issue_move", issueId=a.ref, parentId=ep.lane_id("doing"))
    elif a.cmd == "comment":
        ep.call("epiq_issue_comment_add", issueId=a.ref, body=a.body)
    elif a.cmd == "done":
        if a.body:
            ep.call("epiq_issue_comment_add", issueId=a.ref, body=a.body)
        ep.call("epiq_issue_move", issueId=a.ref, parentId=ep.lane_id("done"))
    elif a.cmd == "list":
        lanes = {l["id"]: l["title"] for l in ep.call("epiq_swimlane_list")}
        for i in sorted(ep.call("epiq_issue_list"), key=lambda i: i.get("createdAt", 0)):
            lane = i.get("swimlane") or lanes.get(i.get("parentNodeId"), "?")
            print(f"{i['id']}  {lane:<12} {i['title']}")
    elif a.cmd == "export":
        export(ep, a.dir)
    elif a.cmd == "call":
        print(json.dumps(ep.call(a.tool, **json.loads(a.json)), indent=1))
    ep.close()


if __name__ == "__main__":
    main()
