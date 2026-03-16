# mesh_utils.py
import os
import gmsh
import meshio
import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import mesh as dmesh


def generate_cantilever_mesh(Lx=2.0, Ly=0.5, Lz=0.5, lc=0.15, filename="mesh.msh"):
    if os.path.exists(filename): os.remove(filename)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cantilever")

    nx = max(1, int(round(Lx / lc)))
    ny = max(1, int(round(Ly / lc)))
    nz = max(1, int(round(Lz / lc)))

    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0)
    l1 = gmsh.model.geo.addLine(p1, p2);
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4);
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(s1)
    gmsh.model.geo.mesh.setRecombine(2, s1)

    gmsh.model.geo.extrude([(2, s1)], 0, 0, Lz, numElements=[nz], recombine=True)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)
    gmsh.finalize()


def load_mesh(filename="mesh.msh"):
    msh = meshio.read(filename)
    hex_data = msh.get_cells_type("hexahedron")
    hex_mesh = meshio.Mesh(points=msh.points, cells=[("hexahedron", hex_data)])
    xdmf_filename = filename.replace(".msh", ".xdmf")
    meshio.write(xdmf_filename, hex_mesh)

    with XDMFFile(MPI.COMM_WORLD, xdmf_filename, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    fdim = domain.topology.dim - 1
    Lx = np.max(domain.geometry.x[:, 0])
    l_bc = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    r_bc = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], Lx))

    indices = np.hstack([l_bc, r_bc])
    values = np.hstack([np.full_like(l_bc, 1, dtype=np.int32), np.full_like(r_bc, 2, dtype=np.int32)])
    sorted_idx = np.argsort(indices)
    return domain, dmesh.meshtags(domain, fdim, indices[sorted_idx], values[sorted_idx])