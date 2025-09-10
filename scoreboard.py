from panda3d.core import LColor
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
from game.constants import TEAM_RED, TEAM_BLUE

class Scoreboard:
    def __init__(self, app):
        self.app = app
        colors = app.cfg.get("colors", {})
        self.team_colors = {
            TEAM_RED: LColor(*colors.get("team_red", (1, 0, 0, 1))),
            TEAM_BLUE: LColor(*colors.get("team_blue", (0, 0, 1, 1))),
        }
        self.visible = False
        self.nodes = []

    def show(self):
        self.visible = True

    def hide(self):
        if not self.visible:
            return
        self.visible = False
        for n in self.nodes:
            n.removeNode()
        self.nodes.clear()

    def update(self, state):
        if not self.visible:
            return
        for n in self.nodes:
            n.removeNode()
        self.nodes.clear()
        if not state:
            return
        players = state.get("players", [])
        teams = state.get("teams", {})
        by_team = {TEAM_RED: [], TEAM_BLUE: []}
        for p in players:
            by_team.setdefault(p.get("team"), []).append(p)
        by_team[TEAM_RED].sort(key=lambda x: x.get("tags", 0), reverse=True)
        by_team[TEAM_BLUE].sort(key=lambda x: x.get("tags", 0), reverse=True)
        y = 0.9
        line = 0.06
        base_x = -1.2
        header = OnscreenText(
            "Name", pos=(base_x, y), align=TextNode.ALeft, fg=(1, 1, 1, 1), scale=0.045
        )
        header_stats = OnscreenText(
            "Ping  Tags  Outs  Cap  Def",
            pos=(base_x + 0.6, y),
            align=TextNode.ALeft,
            fg=(1, 1, 1, 1),
            scale=0.045,
        )
        self.nodes.extend([header, header_stats])
        y -= line
        for team in (TEAM_RED, TEAM_BLUE):
            tname = "RED" if team == TEAM_RED else "BLUE"
            points = teams.get(team, {}).get("captures", 0)
            team_line = OnscreenText(
                f"{tname} - {points}",
                pos=(base_x, y),
                align=TextNode.ALeft,
                fg=self.team_colors[team],
                scale=0.05,
            )
            self.nodes.append(team_line)
            y -= line
            for p in by_team.get(team, []):
                name_color = self.team_colors[team]
                bg = (1, 1, 1, 0.2) if p.get("pid") == self.app.client.pid else (0, 0, 0, 0)
                name_node = OnscreenText(
                    p.get("name", "?"),
                    pos=(base_x, y),
                    align=TextNode.ALeft,
                    fg=name_color,
                    bg=bg,
                    scale=0.045,
                )
                stat_txt = (
                    f"{p.get('ping',0):>4} {p.get('tags',0):>4} {p.get('outs',0):>4} "
                    f"{p.get('captures',0):>4} {p.get('defences',0):>4}"
                )
                stat_node = OnscreenText(
                    stat_txt,
                    pos=(base_x + 0.6, y),
                    align=TextNode.ALeft,
                    fg=(1, 1, 1, 1),
                    scale=0.045,
                )
                self.nodes.extend([name_node, stat_node])
                y -= line
            y -= line * 0.5
