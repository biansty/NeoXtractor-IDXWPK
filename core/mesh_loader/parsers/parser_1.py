import io
from typing import Any, BinaryIO

import numpy as np
from core.binary_readers import read_float, read_uint16, read_uint32, read_uint8
from core.mesh_loader.types import BaseMeshParser, MeshData

class MeshParser1(BaseMeshParser):
    """Standard mesh parser for typical mesh formats with bone hierarchies."""

    def parse(self, data: bytes) -> MeshData:
        """Parse mesh."""
        model = {}

        f = io.BytesIO(data)

        raw_model = self._parse_mesh_original(model, f)
        return self._standardize_mesh_data(raw_model)

    def _parse_mesh_original(self, model: dict[str, Any], f: BinaryIO) -> dict[str, Any]:
        _magic_number = f.read(8)

        # Read mesh version
        current_pos = f.tell()
        f.seek(4)
        model['mesh_version'] = read_uint8(f)

        f.seek(12)
        # Bone count is a little-endian uint16. Reading one byte truncates
        # version 5 skeletons with more than 255 bones.
        model['bone_count'] = read_uint16(f)
        f.seek(current_pos)  # Reset to position after magic number

        model['bone_exist'] = read_uint32(f)
        model['mesh'] = []
        parent_nodes = []

        if model['bone_exist']:
            if model['bone_exist'] > 1:
                count = read_uint8(f)
                f.read(2)
                f.read(count * 4)
            bone_count = read_uint16(f)
            model['bone_count'] = bone_count
            for _ in range(bone_count):
                parent_node = read_uint16(f)
                if parent_node == 65535:
                    parent_node = -1
                parent_nodes.append(parent_node)
            invalid_parents = [
                (bone_index, parent_index)
                for bone_index, parent_index in enumerate(parent_nodes)
                if parent_index != -1 and not 0 <= parent_index < bone_count
            ]
            if invalid_parents:
                raise ValueError(
                    f"Invalid version {model['mesh_version']} bone hierarchy: "
                    f"{invalid_parents[:8]}"
                )
            model['bone_parent'] = parent_nodes

            bone_names = []
            for _ in range(bone_count):
                bone_name = f.read(32)
                bone_name = bone_name.decode(errors='replace').replace('\0', '').replace(' ', '_')
                bone_names.append(bone_name)
            model['bone_name'] = bone_names

            bone_extra_info = read_uint8(f)
            if bone_extra_info:
                for _ in range(bone_count):
                    f.read(28)

            model['bone_matrix'] = []
            for _ in range(bone_count):
                matrix = [read_float(f) for _ in range(16)]
                matrix = np.array(matrix).reshape(4, 4)
                model['bone_matrix'].append(matrix)

        if len(list(filter(lambda x: x == -1, parent_nodes))) > 1:
            num = len(model['bone_parent'])
            model['bone_parent'] = list(map(lambda x: num if x == -1 else x, model['bone_parent']))
            model['bone_parent'].append(-1)
            model['bone_name'].append('dummy_root')
            model['bone_matrix'].append(np.identity(4))

        _flag = read_uint8(f)
        if _flag != 0:
            raise ValueError(f"Unexpected _flag value {_flag} at position {f.tell()}")

        _offset = read_uint32(f)
        while True:
            flag = read_uint16(f)
            if flag == 1:
                break
            f.seek(-2, 1)
            mesh_vertex_count = read_uint32(f)
            mesh_face_count = read_uint32(f)
            _flag = read_uint8(f)
            color_len = read_uint8(f)

            model['mesh'].append((mesh_vertex_count, mesh_face_count, _flag, color_len))

        vertex_count = read_uint32(f)
        face_count = read_uint32(f)

        model['position'] = []
        # vertex position
        for _ in range(vertex_count):
            x = read_float(f)
            y = read_float(f)
            z = read_float(f)
            model['position'].append((x, y, z))

        model['normal'] = []
        # vertex normal
        for _ in range(vertex_count):
            x = read_float(f)
            y = read_float(f)
            z = read_float(f)
            model['normal'].append((x, y, z))

        _flag = read_uint16(f)
        if _flag:
            f.seek(vertex_count * 12, 1)

        model['face'] = []
        # NeoX mesh version 5 widened the index buffer to uint32. This is
        # required for meshes such as 0x8b1b5e7 with 65,691 vertices, which
        # cannot be addressed by a uint16 index. Older versions remain uint16.
        index_reader = read_uint32 if model['mesh_version'] >= 5 else read_uint16
        for _ in range(face_count):
            v1 = index_reader(f)
            v2 = index_reader(f)
            v3 = index_reader(f)
            model['face'].append((v1, v2, v3))

        invalid_faces = [
            (face_index, face)
            for face_index, face in enumerate(model['face'])
            if any(vertex_index >= vertex_count for vertex_index in face)
        ]
        if invalid_faces:
            raise ValueError(
                f"Invalid version {model['mesh_version']} face indices for "
                f"{vertex_count} vertices: {invalid_faces[:8]}"
            )

        model['uv'] = []
        # vertex uv
        for mesh_vertex_count, _, uv_layers, _ in model['mesh']:
            if uv_layers > 0:
                for _ in range(mesh_vertex_count):
                    u = read_float(f)
                    v = read_float(f)
                    model['uv'].append((u, v))
                f.read(mesh_vertex_count * 8 * (uv_layers - 1))
            else:
                for _ in range(mesh_vertex_count):
                    u = 0.0
                    v = 0.0
                    model['uv'].append((u, v))        # vertex color
        for mesh_vertex_count, _, _, color_len in model['mesh']:
            f.read(mesh_vertex_count * 4 * color_len)

        if model['bone_exist']:
            model['vertex_bone'] = []
            for _ in range(vertex_count):
                vertex_bones = [read_uint16(f) for _ in range(4)]
                model['vertex_bone'].append(vertex_bones)

            model['vertex_weight'] = []
            for _ in range(vertex_count):
                vertex_weights = [read_float(f) for _ in range(4)]
                model['vertex_weight'].append(vertex_weights)

        return model
