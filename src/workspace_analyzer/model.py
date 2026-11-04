"""Small, strict URDF model used by all compute and rendering backends."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _vector(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    value = np.asarray(default if text is None else [float(x) for x in text.split()])
    if value.shape != (len(default),) or not np.all(np.isfinite(value)):
        raise ValueError(f"expected {len(default)} finite values, got {text!r}")
    return value


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return a homogeneous transform with URDF roll-pitch-yaw rotation."""
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


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
        if len(set(self.links)) != len(self.links):
            raise ValueError("URDF contains duplicate link names")
        joint_names = [joint.name for joint in self.joints]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("URDF contains duplicate joint names")
        link_names = set(self.links)
        for joint in self.joints:
            if joint.parent not in link_names or joint.child not in link_names:
                raise ValueError(
                    f"joint {joint.name!r} references an unknown parent or child link"
                )
            if joint.parent == joint.child:
                raise ValueError(
                    f"joint {joint.name!r} cannot connect a link to itself"
                )
            if joint.active and (
                not np.isfinite([joint.limit.lower, joint.limit.upper]).all()
                or joint.limit.lower >= joint.limit.upper
            ):
                raise ValueError(
                    f"joint {joint.name!r} requires finite limits with lower < upper"
                )
        self._by_child = {joint.child: joint for joint in self.joints}
        if len(self._by_child) != len(self.joints):
            raise ValueError("each URDF link must have at most one parent joint")
        checked = set()
        for link in self.links:
            visited = set()
            current = link
            while current in self._by_child and current not in checked:
                if current in visited:
                    raise ValueError("URDF joint graph contains a cycle")
                visited.add(current)
                current = self._by_child[current].parent
            checked.update(visited)

    @property
    def root_links(self) -> tuple[str, ...]:
        return tuple(link for link in self.links if link not in self._by_child)

    def zero_transforms(self) -> dict[str, np.ndarray]:
        """Return root-relative link poses with every movable joint set to zero.

        Each root starts at identity. This describes the URDF geometry, even
        when zero is outside a joint's motion limits. Returned arrays are fresh.
        """
        children = {}
        for joint in self.joints:
            children.setdefault(joint.parent, []).append(joint)
        transforms = {root: np.eye(4) for root in self.root_links}
        queue = list(transforms)
        for parent in queue:
            for joint in children.get(parent, ()):
                transforms[joint.child] = transforms[parent] @ joint.origin
                queue.append(joint.child)
        return transforms

    @property
    def leaf_links(self) -> tuple[str, ...]:
        parents = {j.parent for j in self.joints}
        return tuple(link for link in self.links if link not in parents)

    def chain(
        self, base_link: str | None = None, tip_link: str | None = None
    ) -> tuple[Joint, ...]:
        base = base_link or self._default_root()
        if base not in self.links:
            raise ValueError(f"unknown base link {base!r}")
        tip = tip_link or self._default_tip(base)
        if tip not in self.links:
            raise ValueError(f"unknown tip link {tip!r}")
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
        roots = self.root_links
        if len(roots) != 1:
            raise ValueError("base_link is required for a forest URDF")
        return roots[0]

    def _default_tip(self, base: str) -> str:
        """Maximize active-joint count, path length, then lexical tip name."""
        if base not in self.links:
            raise ValueError(f"no kinematic chain starts at {base!r}")
        children = {}
        for joint in self.joints:
            children.setdefault(joint.parent, []).append(joint)
        pending = [(0, 0, base)]
        best = pending[0]
        while pending:
            candidate = pending.pop()
            active_count, length, parent = candidate
            descendants = children.get(parent)
            if not descendants:
                best = max(best, candidate)
                continue
            for joint in descendants:
                pending.append((active_count + joint.active, length + 1, joint.child))
        return best[2]

    @classmethod
    def from_urdf(cls, path: str | Path) -> RobotModel:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            root = ET.parse(source).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"invalid URDF XML in {source}: {exc}") from exc
        if root.tag != "robot":
            raise ValueError("URDF root must be <robot>")
        link_nodes = root.findall("link")
        if any(not node.get("name") for node in link_nodes):
            raise ValueError("URDF link is missing a name")
        links = tuple(node.attrib["name"] for node in link_nodes)
        joints = []
        for node in root.findall("joint"):
            kind = node.attrib.get("type", "fixed")
            name = node.attrib.get("name")
            if not name:
                raise ValueError("URDF joint is missing a name")
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"joint {name!r} has unsupported type {kind!r}")
            if node.find("mimic") is not None:
                raise ValueError(f"joint {name!r} uses unsupported mimic coupling")
            parent_node, child_node = node.find("parent"), node.find("child")
            if (
                parent_node is None
                or child_node is None
                or not parent_node.get("link")
                or not child_node.get("link")
            ):
                raise ValueError(f"joint {name!r} requires parent and child links")
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
            scale = np.max(np.abs(axis))
            if scale == 0 and kind != "fixed":
                raise ValueError(f"joint {name!r} has a zero axis")
            if scale > 0:
                # Normalize before squaring to avoid overflow/underflow for
                # finite axes with very large or very small magnitudes.
                axis = axis / scale
                axis = axis / np.linalg.norm(axis)
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
                    raise ValueError(f"joint {name!r} requires lower/upper limits")
                lower, upper = (
                    float(limit_node.get("lower")),
                    float(limit_node.get("upper")),
                )
            velocity = (
                None
                if limit_node is None or limit_node.get("velocity") is None
                else float(limit_node.get("velocity"))
            )
            if velocity is not None and (not np.isfinite(velocity) or velocity <= 0):
                raise ValueError(f"joint {name!r} requires a positive finite velocity")
            joints.append(
                Joint(
                    name,
                    kind,
                    parent_node.attrib["link"],
                    child_node.attrib["link"],
                    origin,
                    axis,
                    JointLimit(lower, upper, velocity),
                )
            )
        if not links:
            raise ValueError("URDF has no links")
        return cls(root.attrib.get("name", source.stem), links, tuple(joints), source)
