# mesh_utils.py
import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmsh_io

def generate_cantilever_mesh(Lx=2.0, Ly=0.5, Lz=0.5, lc=0.2, filename="mesh.msh"):
    """生成悬臂梁网格并保存为 .msh 文件"""
    gmsh.initialize()
    gmsh.model.add("cantilever_beam")
    gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()
    # 获取左右端面并设置物理组
    surfaces = gmsh.model.getEntities(2)
    left_tag = right_tag = None
    for surf in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        if np.isclose(com[0], 0.0):
            left_tag = surf[1]
        elif np.isclose(com[0], Lx):
            right_tag = surf[1]
    if left_tag is None or right_tag is None:
        raise RuntimeError("无法找到左端面或右端面")
    gmsh.model.addPhysicalGroup(2, [left_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "fixed")
    gmsh.model.addPhysicalGroup(2, [right_tag], tag=2)
    gmsh.model.setPhysicalName(2, 2, "load")
    volumes = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=3)
    gmsh.model.setPhysicalName(3, 3, "domain")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"网格已保存至 {filename}")

def load_mesh(filename="mesh.msh"):
    """从 .msh 文件加载网格，返回 domain 和 facet_tags"""
    mesh_data = gmsh_io.read_from_msh(filename, MPI.COMM_WORLD, gdim=3)
    domain = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    return domain, facet_tags