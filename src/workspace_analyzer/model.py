"""Small, strict URDF model used by all compute and rendering backends."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _vector(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray(default if text is None else [float(x) for x in text.split()])


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(r),
        np.sin(r),
        np.cos(p),
        np.sin(p),
        np.cos(y),
        np.sin(y),
    )
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rot
    return out


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float
    velocity: float | None = None


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limit: JointLimit

    @property
    def active(self) -> bool:
        return self.kind in {"revolute", "continuous", "prismatic"}


@dataclass
class RobotModel:
    name: str
    links: tuple[str, ...]
    joints: tuple[Joint, ...]
    source: Path | None = None
    _by_child: dict[str, Joint] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_child = {joint.child: joint for joint in self.joints}
        if len(self._by_child) != len(self.joints):
            raise ValueError("each URDF link must have at most one parent joint")

    @property
    def root_links(self) -> tuple[str, ...]:
        children = set(self._by_child)
        return tuple(link for link in self.links if link not in children)

    @property
    def leaf_links(self) -> tuple[str, ...]:
        parents = {j.parent for j in self.joints}
        return tuple(link for link in self.links if link not in parents)

    def chain(
        self, base_link: str | None = None, tip_link: str | None = None
    ) -> tuple[Joint, ...]:
        base = base_link or self._default_root()
        tip = tip_link or self._default_tip(base)
        result: list[Joint] = []
        current = tip
        while current != base:
            joint = self._by_child.get(current)
            if joint is None:
                raise ValueError(f"{tip!r} is not below base link {base!r}")
            result.append(joint)
            current = joint.parent
        result.reverse()
        return tuple(result)

    def _default_root(self) -> str:
        if len(self.root_links) != 1:
            raise ValueError("base_link is required for a forest URDF")
        return self.root_links[0]

    def _default_tip(self, base: str) -> str:
        candidates = []
        for leaf in self.leaf_links:
            try:
                chain = self.chain(base, leaf)
            except ValueError:
                continue
            candidates.append((sum(j.active for j in chain), len(chain), leaf))
        if not candidates:
            raise ValueError(f"no kinematic chain starts at {base!r}")
        return max(candidates)[2]

    @classmethod
    def from_urdf(cls, path: str | Path) -> RobotModel:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        root = ET.parse(source).getroot()
        if root.tag != "robot":
            raise ValueError("URDF root must be <robot>")
        links = tuple(node.attrib["name"] for node in root.findall("link"))
        joints = []
        for node in root.findall("joint"):
            kind = node.attrib.get("type", "fixed")
            origin_node = node.find("origin")
            xyz = _vector(
                None if origin_node is None else origin_node.get("xyz"), (0.0, 0.0, 0.0)
            )
            rpy = _vector(
                None if origin_node is None else origin_node.get("rpy"), (0.0, 0.0, 0.0)
            )
            origin = _rpy_matrix(rpy)
            origin[:3, 3] = xyz
            axis_node = node.find("axis")
            axis = _vector(
                None if axis_node is None else axis_node.get("xyz"), (1.0, 0.0, 0.0)
            )
            norm = np.linalg.norm(axis)
            if norm == 0 and kind != "fixed":
                raise ValueError(f"joint {node.attrib['name']!r} has a zero axis")
            axis = axis if norm == 0 else axis / norm
            limit_node = node.find("limit")
            if kind == "continuous":
                lower, upper = -np.pi, np.pi
            elif kind == "fixed":
                lower = upper = 0.0
            else:
                if (
                    limit_node is None
                    or "lower" not in limit_node.attrib
                    or "upper" not in limit_node.attrib
                ):
                    raise ValueError(
                        f"joint {node.attrib['name']!r} requires lower/upper limits"
                    )
                lower, upper = (
                    float(limit_node.get("lower")),
                    float(limit_node.get("upper")),
                )
            velocity = (
                None
                if limit_node is None or limit_node.get("velocity") is None
                else float(limit_node.get("velocity"))
            )
            joints.append(
                Joint(
                    node.attrib["name"],
                    kind,
                    node.find("parent").attrib["link"],
                    node.find("child").attrib["link"],
                    origin,
                    axis,
                    JointLimit(lower, upper, velocity),
                )
            )
        if not links:
            raise ValueError("URDF has no links")
        return cls(root.attrib.get("name", source.stem), links, tuple(joints), source)
