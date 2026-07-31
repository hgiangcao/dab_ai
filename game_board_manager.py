from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Point:
    id: int
    row: int
    col: int

    edges: list = field(default_factory=list)
    degree: int = 0


@dataclass
class Edge:
    id: int
    p1: Point
    p2: Point

    boxes: list = field(default_factory=list)
    occupied: bool = False
    direction: str = ""   # "horizontal" or "vertical"


@dataclass
class Box:
    id: int
    row: int
    col: int

    edges: dict = field(default_factory=dict)   # "top"/"bottom"/"left"/"right" -> Edge
    points: dict = field(default_factory=dict)  # corners (optional future use)

    edge_count: int = 0
    owner: Optional[int] = None


class Board_Manager:

    def __init__(self, rows=5, cols=5):

        self.rows = rows
        self.cols = cols

        self.points = [
            [
                Point(r * (cols + 1) + c, r, c)
                for c in range(cols + 1)
            ]
            for r in range(rows + 1)
        ]

        self.boxes = []
        self.edges = []

        # Edge matrices for O(1) lookup: h_edges[r][c], v_edges[r][c]
        self.h_edges = [[None] * cols for _ in range(rows + 1)]
        self.v_edges = [[None] * (cols + 1) for _ in range(rows)]

        # Quick bucket query: box_by_edge_count[n] = set of box IDs with n occupied edges
        self.box_by_edge_count = [
            set(),   # 0-edge box
            set(),   # 1-edge box
            set(),   # 2-edge box
            set(),   # 3-edge box
            set()    # 4-edge box (complete)
        ]

        self.create_board()

        # Set of unoccupied edge IDs — updated by apply_move / undo_move
        self.available_edges = set(range(len(self.edges)))

    # ------------------------------------------------------------------
    # Board construction
    # ------------------------------------------------------------------

    def create_board(self):
        # 1. Create boxes
        for r in range(self.rows):
            for c in range(self.cols):
                box = Box(id=len(self.boxes), row=r, col=c)
                self.boxes.append(box)
                self.box_by_edge_count[0].add(box.id)

        # 2. Create horizontal edges and connect to boxes
        #    h_edges[r][c] is between points[r][c] and points[r][c+1]
        #    - top  edge of box (r,   c) if r < rows
        #    - bottom edge of box (r-1, c) if r > 0
        for r in range(self.rows + 1):
            for c in range(self.cols):
                edge = self._create_edge(self.points[r][c], self.points[r][c + 1], "horizontal")
                self.h_edges[r][c] = edge
                if r < self.rows:
                    self._connect_box_edge(self.boxes[r * self.cols + c], edge, "top")
                if r > 0:
                    self._connect_box_edge(self.boxes[(r - 1) * self.cols + c], edge, "bottom")

        # 3. Create vertical edges and connect to boxes
        #    v_edges[r][c] is between points[r][c] and points[r+1][c]
        #    - left  edge of box (r, c)   if c < cols
        #    - right edge of box (r, c-1) if c > 0
        for r in range(self.rows):
            for c in range(self.cols + 1):
                edge = self._create_edge(self.points[r][c], self.points[r + 1][c], "vertical")
                self.v_edges[r][c] = edge
                if c < self.cols:
                    self._connect_box_edge(self.boxes[r * self.cols + c], edge, "left")
                if c > 0:
                    self._connect_box_edge(self.boxes[r * self.cols + (c - 1)], edge, "right")

    def _create_edge(self, p1: Point, p2: Point, direction: str) -> Edge:
        edge = Edge(id=len(self.edges), p1=p1, p2=p2, direction=direction)
        self.edges.append(edge)
        p1.edges.append(edge)
        p2.edges.append(edge)
        return edge

    def _connect_box_edge(self, box: Box, edge: Edge, side: str):
        """Wire an edge to a box bidirectionally."""
        box.edges[side] = edge
        edge.boxes.append(box)

    # ------------------------------------------------------------------
    # Move application / undo  (self-contained — no external IDs needed)
    # ------------------------------------------------------------------

    def apply_move(self, edge_id: int, current_player: int) -> List[int]:
        """
        Mark an edge as occupied and update all adjacent boxes.
        Uses edge.boxes directly — no external box IDs required.

        Args:
            edge_id:        Index into self.edges.
            current_player: The player drawing the line (1 or -1).

        Returns:
            List of box IDs that were just completed (edge_count reached 4).
        """
        edge = self.edges[edge_id]
        edge.occupied = True
        edge.p1.degree += 1
        edge.p2.degree += 1
        self.available_edges.discard(edge_id)

        newly_captured = []
        for box in edge.boxes:
            self.box_by_edge_count[box.edge_count].discard(box.id)
            box.edge_count += 1
            self.box_by_edge_count[box.edge_count].add(box.id)
            if box.edge_count == 4:
                box.owner = current_player
                newly_captured.append(box.id)

        return newly_captured

    def undo_move(self, edge_id: int):
        """
        Reverse the effect of apply_move.
        Boxes that were captured (edge_count == 4) are auto-detected and cleared.

        Args:
            edge_id: Index into self.edges (same value passed to apply_move).
        """
        edge = self.edges[edge_id]
        edge.occupied = False
        edge.p1.degree -= 1
        edge.p2.degree -= 1
        self.available_edges.add(edge_id)

        for box in edge.boxes:
            self.box_by_edge_count[box.edge_count].discard(box.id)
            was_captured = (box.edge_count == 4)
            box.edge_count -= 1
            self.box_by_edge_count[box.edge_count].add(box.id)
            if was_captured:
                box.owner = None

    # ------------------------------------------------------------------
    # Bucket / structural queries
    # ------------------------------------------------------------------

    def boxes_with_n_edges(self, n: int) -> set:
        """Return the set of box IDs that currently have exactly n edges occupied."""
        return self.box_by_edge_count[n]

    def open_boxes(self) -> set:
        """Return box IDs not yet completed (< 4 edges)."""
        result = set()
        for i in range(4):
            result |= self.box_by_edge_count[i]
        return result

    def completed_boxes(self) -> set:
        """Return box IDs that are fully captured (4 edges)."""
        return self.box_by_edge_count[4]

    # ------------------------------------------------------------------
    # Isolated-edge queries
    # ------------------------------------------------------------------

    def is_isolated_edge(self, edge_id: int) -> bool:
        """
        True if the edge is unoccupied AND both its endpoints have degree 0
        (i.e. no other edge touches those points yet).
        Useful for safe early-game moves.
        """
        edge = self.edges[edge_id]
        return (not edge.occupied
                and edge.p1.degree == 0
                and edge.p2.degree == 0)

    def get_isolated_moves(self) -> List[int]:
        """
        Return IDs of all currently available edges that are isolated
        (both endpoints untouched). O(available_edges).
        """
        return [e_id for e_id in self.available_edges
                if self.is_isolated_edge(e_id)]