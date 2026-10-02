"""PMX exporter adapted from zhouhang95/neox_tools.

Exports NeoX coordinates as PMX coordinates (-X, Y, -Z), preserves the
skeleton hierarchy, and writes up to four normalized skin influences.
"""

from __future__ import annotations

import io
import math
from collections import defaultdict
from typing import Iterable

from pymeshio import common, pmx
import pymeshio.pmx.writer

from core.mesh_loader import MeshData

NAME = "Polygon Model eXtended (PMX) Format"
EXTENSION = ".pmx"

_INVALID_JOINTS = {-1, 0xFF, 0xFFFF, 0xFFFFFFFF}
_EPSILON = 1.0e-8


def _finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def _vec3(values: Iterable[float], default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    values = list(values)
    return tuple(_finite(values[i], default[i]) if i < len(values) else default[i] for i in range(3))


def _to_pmx_position(position) -> common.Vector3:
    """Match neox_tools: NeoX (x,y,z) -> PMX (-x,y,-z)."""
    x, y, z = _vec3(position)
    return common.Vector3(-x, y, -z)


def _to_pmx_normal(normal) -> common.Vector3:
    x, y, z = _vec3(normal, (0.0, 0.0, 1.0))
    return common.Vector3(-x, y, -z)


def _matrix_translation(matrix) -> tuple[float, float, float]:
    """Equivalent to transformations.translation_from_matrix(matrix.T).

    NeoX stores the translation in the last row of the original bind matrix.
    A fallback to the last column supports parsers that already transposed it.
    """
    try:
        row = _vec3((matrix[3, 0], matrix[3, 1], matrix[3, 2]))
        col = _vec3((matrix[0, 3], matrix[1, 3], matrix[2, 3]))
    except (IndexError, TypeError):
        return 0.0, 0.0, 0.0
    if any(abs(value) > _EPSILON for value in row):
        return row
    return col


def _unique_bone_names(names: list[str], count: int) -> list[str]:
    used: set[str] = set()
    result: list[str] = []
    for index in range(count):
        raw = names[index] if index < len(names) else ""
        base = str(raw or f"bone_{index}").replace("\0", "").strip() or f"bone_{index}"
        name = base
        suffix = 1
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        result.append(name)
    return result


def _order_bones(parents: list[int]) -> tuple[list[int], dict[int, int]]:
    """Depth-first order, like neox_tools, but includes every root and orphan."""
    count = len(parents)
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []
    for index, parent in enumerate(parents):
        if parent == -1 or not 0 <= parent < count or parent == index:
            roots.append(index)
        else:
            children[parent].append(index)

    order: list[int] = []
    visited: set[int] = set()

    def visit(index: int) -> None:
        if index in visited:
            return
        visited.add(index)
        order.append(index)
        for child in children.get(index, []):
            visit(child)

    for root in roots:
        visit(root)
    for index in range(count):
        visit(index)
    return order, {old: new for new, old in enumerate(order)}


def _build_bones(mesh: MeshData) -> tuple[list[pmx.Bone], dict[int, int]]:
    count = min(len(mesh.bone_parent), len(mesh.bone_name), len(mesh.bone_matrix))
    if count <= 0:
        root = pmx.Bone(
            "root", "root", common.Vector3(0.0, 0.0, 0.0), -1, 0, 0,
            tail_position=common.Vector3(0.0, 1.0, 0.0),
        )
        for flag in (
            pmx.BONEFLAG_CAN_ROTATE,
            pmx.BONEFLAG_CAN_TRANSLATE,
            pmx.BONEFLAG_IS_VISIBLE,
            pmx.BONEFLAG_CAN_MANIPULATE,
        ):
            root.setFlag(flag, True)
        return [root], {0: 0}

    parents = [int(value) for value in mesh.bone_parent[:count]]
    names = _unique_bone_names(list(mesh.bone_name), count)
    order, old_to_new = _order_bones(parents)
    children: dict[int, list[int]] = defaultdict(list)
    for child, parent in enumerate(parents):
        if 0 <= parent < count and parent != child:
            children[parent].append(child)

    bones: list[pmx.Bone] = []
    for old_index in order:
        old_parent = parents[old_index]
        new_parent = old_to_new.get(old_parent, -1)
        new_index = old_to_new[old_index]
        if new_parent >= new_index:
            new_parent = -1

        position = _to_pmx_position(_matrix_translation(mesh.bone_matrix[old_index]))
        child_candidates = children.get(old_index, [])
        tail_index = old_to_new[child_candidates[0]] if child_candidates else -1
        bone = pmx.Bone(
            name=names[old_index],
            english_name=names[old_index],
            position=position,
            parent_index=new_parent,
            layer=0,
            flag=0,
            tail_position=common.Vector3(0.0, 1.0, 0.0),
            tail_index=tail_index,
        )
        if tail_index >= 0:
            bone.setFlag(pmx.BONEFLAG_TAILPOS_IS_BONE, True)
        bone.setFlag(pmx.BONEFLAG_CAN_ROTATE, True)
        bone.setFlag(pmx.BONEFLAG_IS_VISIBLE, True)
        bone.setFlag(pmx.BONEFLAG_CAN_MANIPULATE, True)
        if new_parent == -1:
            bone.setFlag(pmx.BONEFLAG_CAN_TRANSLATE, True)
        bones.append(bone)
    return bones, old_to_new


def _vertex_deform(mesh: MeshData, vertex_index: int, old_to_new: dict[int, int]):
    joints = mesh.vertex_bone[vertex_index] if vertex_index < len(mesh.vertex_bone) else []
    weights = mesh.vertex_weight[vertex_index] if vertex_index < len(mesh.vertex_weight) else []
    merged: dict[int, float] = defaultdict(float)

    for raw_joint, raw_weight in zip(joints, weights):
        try:
            old_joint = int(raw_joint)
            weight = max(0.0, _finite(raw_weight))
        except (TypeError, ValueError, OverflowError):
            continue
        if old_joint in _INVALID_JOINTS or weight <= _EPSILON:
            continue
        new_joint = old_to_new.get(old_joint)
        if new_joint is not None:
            merged[new_joint] += weight

    influences = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:4]
    if not influences:
        return pmx.Bdef1(0)
    total = sum(weight for _, weight in influences)
    if total <= _EPSILON:
        return pmx.Bdef1(influences[0][0])
    influences = [(joint, weight / total) for joint, weight in influences]

    if len(influences) == 1 or influences[0][1] >= 1.0 - _EPSILON:
        return pmx.Bdef1(influences[0][0])
    if len(influences) == 2:
        return pmx.Bdef2(influences[0][0], influences[1][0], influences[0][1])
    while len(influences) < 4:
        influences.append((influences[0][0], 0.0))
    return pmx.Bdef4(
        *(joint for joint, _ in influences),
        *(weight for _, weight in influences),
    )


def _add_material(model: pmx.Model, index: int, index_count: int) -> None:
    model.materials.append(pmx.Material(
        name=f"Mat{index}",
        english_name=f"material{index}",
        diffuse_color=common.RGB(1.0, 1.0, 1.0),
        alpha=1.0,
        specular_factor=1.0,
        specular_color=common.RGB(1.0, 1.0, 1.0),
        ambient_color=common.RGB(0.0, 0.0, 0.0),
        flag=0,
        edge_color=common.RGBA(0.0, 0.0, 0.0, 1.0),
        edge_size=0.0,
        texture_index=-1,
        sphere_texture_index=-1,
        sphere_mode=pmx.MATERIALSPHERE_NONE,
        toon_sharing_flag=1,
        toon_texture_index=0,
        comment="Auto-generated material",
        vertex_count=index_count,
    ))


def convert(mesh: MeshData) -> bytes:
    """Convert a parsed NeoX mesh to a Blender/MMD-compatible PMX file."""
    model = pmx.Model(version=2.0)
    model.name = "NeoX Model"
    model.english_name = "NeoX Model"
    model.comment = "NeoX Model Converterで生成"
    model.english_comment = "Created by NeoX Model Converter."

    model.bones, old_to_new = _build_bones(mesh)
    model.display_slots.append(
        pmx.DisplaySlot("骨", "Bones", 0, [(0, index) for index in range(len(model.bones))])
    )
    model.display_slots.append(pmx.DisplaySlot("表情", "Exp", 1, []))

    for index, position in enumerate(mesh.position):
        normal = mesh.normal[index] if index < len(mesh.normal) else (0.0, 0.0, 1.0)
        uv = mesh.uv[index] if index < len(mesh.uv) else (0.0, 0.0)
        u = _finite(uv[0]) if len(uv) > 0 else 0.0
        v = _finite(uv[1]) if len(uv) > 1 else 0.0
        deform = _vertex_deform(mesh, index, old_to_new) if mesh.has_bones else pmx.Bdef1(0)
        model.vertices.append(pmx.Vertex(
            _to_pmx_position(position),
            _to_pmx_normal(normal),
            common.Vector2(u, v),
            deform,
            0.0,
        ))

    vertex_count = len(model.vertices)
    for face in mesh.face:
        if len(face) == 3 and all(0 <= int(vertex) < vertex_count for vertex in face):
            # Matches neox_tools PMX export. Coordinate reflection already flips winding.
            model.indices.extend(int(vertex) for vertex in face)

    available = len(model.indices)
    material_counts: list[int] = []
    for item in mesh.mesh:
        face_count = int(item[1])
        count = min(max(face_count * 3, 0), available - sum(material_counts))
        if count > 0:
            material_counts.append(count)
    remainder = available - sum(material_counts)
    if remainder > 0:
        material_counts.append(remainder)
    if not material_counts:
        material_counts = [available]
    for index, count in enumerate(material_counts):
        _add_material(model, index, count)

    output = io.BytesIO()
    pymeshio.pmx.writer.write(output, model)
    return output.getvalue()
