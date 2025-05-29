import torch
import os
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.io import load_objs_as_meshes
from scene.cameras import ConvertGS2Mesh
from PIL import Image
import nvdiffrast.torch as dr

Image.MAX_IMAGE_PIXELS = None

def get_obj_files(mesh_path):
    obj_files = []
    for root, dirs, files in os.walk(mesh_path):
        for file in files:
            if file.endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    return obj_files

class MeshRender:
    def __init__(self, mesh_path):
        self.device = 'cuda'
        self.mesh_path = mesh_path
        self.use_opengl = False
        
        obj_files = get_obj_files(self.mesh_path)
        self.textured_mesh = join_meshes_as_scene(load_objs_as_meshes(obj_files).to("cuda"))
        
        self.uv = self.textured_mesh.textures.verts_uvs_list()[0].float()
        self.uv_idx = self.textured_mesh.textures.faces_uvs_list()[0].int()
        self.tex = self.textured_mesh.textures.maps_padded()
        self.faces = self.textured_mesh.faces_list()[0].int()
        self.verts = self.textured_mesh.verts_list()[0].float()
        self.glctx = dr.RasterizeGLContext() if self.use_opengl else dr.RasterizeCudaContext()
        
        
    def render(self, camera):
        # Get full projection matrix
        camera_mtx = camera.full_proj_transform_opengl
                
        # Convert to homogeneous coordinates
        pos = torch.cat([self.verts, torch.ones([self.verts.shape[0], 1], device=self.device)], axis=1)
                
        # Transform points to NDC/clip space
        pos = torch.matmul(pos, camera_mtx)[None]
        pos[..., :3] = -pos[..., :3]
        
        # Rasterize with NVDiffRast
        rast_out, _ = dr.rasterize(self.glctx, pos=pos, tri=self.faces, resolution=[camera.image_height, camera.image_width])
        zbuf = rast_out[..., 2]
        texc, _ = dr.interpolate(self.uv[None, ...], rast_out, self.uv_idx)
        texc[0, :, :, 1] = 1 - texc[0, :, :, 1]
        color = dr.texture(self.tex, texc, filter_mode='linear')
        color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
        zbuf = torch.where(zbuf < 0, 1/-zbuf, -1)
        return color[0].permute(2, 0, 1), zbuf
    
    
    