"""
Dataloader Class
    Based of nerfacc
"""

import collections
import os
import numpy as np
import torch
import torch.nn.functional as F
import models.data_utils as du

Rays = collections.namedtuple("Rays", ("origins", "viewdirs", "image_id"))


class DataLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str = '',
        root_fp: str = '',
        color_bkgd_aug: str = "black",
        device: str = '',
        args: dict = None,
        isTest: bool = False,
        nadir_height: int = 100
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black",
                                  "random"], "Invaild Background color"
        if subject_id == '':
            subject_id = args['scene']
        if root_fp == '':
            root_fp = args['data_root_fp']
        device = args['device'] if device == '' else 'cpu'

        self.num_rays = args['target_sample_batch_size'] // args['render_n_samples']
        self.color_bkgd_aug = color_bkgd_aug
        self.args = args
        # Use fixed height for nadir rays, seems to work better with lower heights (e.g. 100 meters)
        self.nadir_height = nadir_height
        self.isTest = isTest

        if self.isTest:
            self.c2w, self.K, self.WIDTH, self.HEIGHT, self.aabb, self.num_train_img, self.enu_center = \
                du.generateTestPoses(
                    root_fp, subject_id, args['num_of_frames'], args['downscale_factor'])

        else:
            if self.args['depth_fraction']:
                data_dir = os.path.join(root_fp, subject_id)
                transforms_path = os.path.join(data_dir, 'transforms.json')
                reproj_path = os.path.join(data_dir, 'reproj_err.json')
                self.aabb, self.K, self.c2w, self.images, self.depths = \
                    du.read_reproj_jsons(transforms_path, reproj_path)

                self.num_train_img = self.images.shape[0]
                self.max_reproj_err = self.depths[:, -1].max()
            else:
                self.images, self.depths, self.c2w, self.K, self.aabb, self.num_train_img = \
                    du.load_renderings(root_fp, subject_id)

            self.images = torch.from_numpy(
                self.images).to(torch.uint8).to(device)
            self.depths = torch.from_numpy(
                self.depths).to(torch.float32).to(device)

            self.HEIGHT, self.WIDTH = self.images.shape[1:3]

            if self.args['patch_padding']:
                patch_len = self.args['patch_padding']*2+1
                # Create a sliding window of patches (all combinations) and cache it
                self.all_patches = torch.arange(self.HEIGHT*self.WIDTH, device=self.images.device). \
                    view(self.HEIGHT, self.WIDTH).unfold(0, patch_len, 1).unfold(
                        1, patch_len, 1).reshape(-1, patch_len, patch_len)

                self.novel_c2w, self.novel_K, _, _, _, _, _ = du.generateTestPoses(
                    root_fp, subject_id, self.num_train_img, 1)
                self.novel_c2w = torch.from_numpy(
                    self.novel_c2w).to(torch.float32).to(device)
                self.novel_K = torch.from_numpy(
                    self.novel_K).to(torch.float32).to(device)

        self.c2w = torch.from_numpy(self.c2w).to(torch.float32).to(device)
        self.K = torch.from_numpy(self.K).to(torch.float32).to(device)
        self.aabb = torch.tensor(self.aabb, dtype=torch.float32, device=device)
        # Average z height as a correction factor
        self.avg_z = self.c2w[:, 2, 3].mean()

    # =======================================================================================================================================
    # General class methods

    def __len__(self):
        return len(self.c2w)

    @torch.no_grad()
    def __getitem__(self, index):
        if self.isTest:
            data = self.fetch_test_data(index)
        else:
            if self.args['patch_padding']:
                data = self.fetch_data_novel_patch(
                    self.args['patch_padding'], 0.5, index)
            elif self.args['depth_fraction']:
                data = self.fetch_data_depth(
                    self.args['depth_fraction'], index)
            else:
                data = self.fetch_data(index)

            data = self.preprocess(data)
        return data

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def get_average_zval(self):
        return self.avg_z

    # =======================================================================================================================================
    # Test class methods

    def createNadirRays(self, samp_dist=0.50):
        # every pixel in the image corresponds to 15 cm
        stepsX = int((self.aabb[3] - self.aabb[0]) / samp_dist)
        stepsY = int((self.aabb[4] - self.aabb[1]) / samp_dist)

        xs = torch.linspace(self.aabb[0], self.aabb[3], steps=stepsX, device=self.c2w.device)
        ys = torch.linspace(-self.aabb[1], -self.aabb[4], steps=stepsY, device=self.c2w.device)

        x, y = torch.meshgrid(xs, ys, indexing="xy")
        x = x.flatten()
        y = y.flatten()

        # all rays have same z value, from camera center
        z = torch.ones([x.shape[0],], device=self.c2w.device) * self.nadir_height

        # every ray has a different origin
        origins = torch.stack([x, y, z], dim=-1)

        # all rays have same view directions [0,0,-1] because all are ortagonal to surface
        viewdirs = torch.broadcast_to(torch.tensor(
            [0.0, 0.0, -1.0], device=self.c2w.device), origins.shape)

        # Convert rays into [h, w, 3] format
        origins = torch.reshape(origins, (stepsY, stepsX, 3))
        viewdirs = torch.reshape(viewdirs, (stepsY, stepsX, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=None)
        color_bkgd = torch.zeros(3, device=self.c2w.device)

        return {
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": color_bkgd
        }

    def fetch_test_data(self, idx, downscale_factor=1):
        """Fetch the data (it maybe cached for multiple batches)."""

        test_WIDTH = int(self.WIDTH / downscale_factor)
        test_HEIGHT = int(self.HEIGHT / downscale_factor)

        x, y = torch.meshgrid(
            torch.linspace(0, self.WIDTH, steps=test_WIDTH,
                           device=self.c2w.device),
            torch.linspace(0, self.HEIGHT, steps=test_HEIGHT,
                           device=self.c2w.device),
            indexing="xy",
        )

        x = x.flatten()
        y = y.flatten()

        # generate rays
        c2w = self.c2w[[idx]]  # (num_rays, 3, 4)
        K = self.K
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[idx, 0, 2] + 0.5) / K[idx, 0, 0],
                    (y - K[idx, 1, 2] + 0.5) / K[idx, 1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        origins = torch.reshape(origins, (test_HEIGHT, test_WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (test_HEIGHT, test_WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=None)

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.c2w.device)

        return {
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": color_bkgd
        }

    def get_rays(self, c2w):
        """Generates a custom set of rays from given c2w transformation matrix. Uses perspective camera model.

        Args:
            c2w (4x4 Numpy Array): Camera pose matrix to originate rays

        Returns:
            Data (obj): Output object including the Rays and the background color for rendering 
        """
        c2w = torch.tensor(c2w, dtype=torch.float, device=self.c2w.device)

        x, y = torch.meshgrid(
            torch.arange(self.WIDTH, device=self.c2w.device),
            torch.arange(self.HEIGHT, device=self.c2w.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        K = self.K
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[0, 2] + 0.5) / K[0, 0],
                    (y - K[1, 2] + 0.5) / K[1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=None)

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.c2w.device)

        return {
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": color_bkgd
        }

    def get_nadir_rays(self):
        """Generate WxH nadir rays centered at dataset aabb coordinates.
        When downscale factor is large (or W and H are small) this function generates more sparse rays

        Returns:
            Data (obj): Output object including the Rays and the background color for rendering 
        """
        x = np.array([self.aabb[0].item(), self.aabb[3].item()])
        y = x * (self.HEIGHT/self.WIDTH)

        xs = torch.linspace(x[0], x[1], steps=self.WIDTH,
                            device=self.c2w.device)
        ys = torch.linspace(-y[0], -y[1], steps=self.HEIGHT,
                            device=self.c2w.device)

        x, y = torch.meshgrid(xs, ys, indexing="xy")
        x = x.flatten()
        y = y.flatten()

        # all rays have same z value, from camera center
        z = torch.ones([x.shape[0],], device=self.c2w.device) * \
            self.nadir_height

        # every ray has a different origin
        origins = torch.stack([x, y, z], dim=-1)

        # all rays have same view directions [0,0,-1] because all are ortagonal to surface
        viewdirs = torch.broadcast_to(torch.tensor(
            [0.0, 0.0, -1.0], device=self.c2w.device), origins.shape)

        # Convert rays into [h, w, 3] format
        origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=None)

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.c2w.device)

        return {
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": color_bkgd
        }

    # =======================================================================================================================================
    # Train class methods

    def fetch_data(self, index):

        num_rays = self.num_rays
        image_id = torch.randint(0, len(self.images), size=(
            num_rays,), device=self.images.device)

        x = torch.randint(0, self.WIDTH, size=(
            num_rays,), device=self.images.device)
        y = torch.randint(0, self.HEIGHT, size=(
            num_rays,), device=self.images.device)

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.c2w[image_id]  # (num_rays, 3, 4)
        K = self.K[image_id]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:, 0, 2] + 0.5) / K[:, 0, 0],
                    (y - K[:, 1, 2] + 0.5) / K[:, 1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / \
            torch.linalg.norm(directions, dim=-1, keepdims=True)

        origins = torch.reshape(origins, (num_rays, 3))
        viewdirs = torch.reshape(viewdirs, (num_rays, 3))
        rgba = torch.reshape(rgba, (num_rays, rgba.shape[1]))

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=image_id)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }

    def fetch_data_patch(self, patch_padding, index):
        # patch_padding: number of pixels on each side to include for the window
        #  e.g. patch_padding=1, 3x3 window, patch_padding=2, 5x5 window
        patch_len = patch_padding*2+1
        num_patches = self.num_rays//patch_len**2
        num_rays = num_patches*patch_len**2
        image_id = torch.randint(0, len(self.images), size=(
            num_patches,), device=self.images.device)
        image_id = image_id.expand(patch_len**2, -1).T.flatten()

        # Get patches from all possible windows
        ind = torch.randint(0, self.all_patches.shape[0], size=(
            num_patches,), device=self.images.device)
        pixel_nums = self.all_patches[ind, ...].view(-1)
        x = pixel_nums % self.WIDTH
        y = pixel_nums // self.WIDTH

        # x, y = [], []
        # for k in range(num_patches):
        #     for j in range(patch_len):
        #         for i in range(patch_len):
        #             x.append(xc[k]+i)
        #             y.append(yc[k]+j)

        # x = torch.stack(x).to(device=self.images.device)
        # y = torch.stack(y).to(device=self.images.device)

        # xy = torch.vstack((x, y)).T # debug

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.c2w[image_id]  # (num_rays, 3, 4)
        K = self.K[image_id]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:, 0, 2] + 0.5) / K[:, 0, 0],
                    (y - K[:, 1, 2] + 0.5) / K[:, 1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / \
            torch.linalg.norm(directions, dim=-1, keepdims=True)

        origins = torch.reshape(origins, (num_rays, 3))
        viewdirs = torch.reshape(viewdirs, (num_rays, 3))
        rgba = torch.reshape(rgba, (num_rays, rgba.shape[1]))

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=image_id)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }

    def fetch_data_novel_patch(self, patch_padding, fraction, index):
        # patch_padding: number of pixels on each side to include for the window
        #  e.g. patch_padding=1, 3x3 window, patch_padding=2, 5x5 window
        patch_len = patch_padding*2+1
        num_patches = int(self.num_rays / patch_len**2 *
                          fraction)  # rays for novel patch
        # rays for regular training
        num_tr_rays = int(self.num_rays - num_patches*patch_len**2)

        # Get regular rays
        image_id = torch.randint(0, len(self.images), size=(
            self.num_rays,), device=self.images.device)
        x = torch.randint(0, self.WIDTH, size=(
            num_tr_rays,), device=self.images.device)
        y = torch.randint(0, self.HEIGHT, size=(
            num_tr_rays,), device=self.images.device)
        rgba = self.images[image_id[:num_tr_rays], y, x] / \
            255.0  # (num_rays, 4)

        # Get patches from all possible windows
        ind = torch.randint(0, self.all_patches.shape[0], size=(
            num_patches,), device=self.images.device)
        pixel_nums = self.all_patches[ind, ...].view(-1)
        x = torch.hstack((x, pixel_nums % self.WIDTH))
        y = torch.hstack((y, pixel_nums // self.WIDTH))

        # generate rays
        # (num_rays, 3, 4)
        c2w = torch.vstack(
            (self.c2w[image_id[:num_tr_rays]], self.novel_c2w[image_id[num_tr_rays:]]))
        K = torch.vstack((self.K[image_id[:num_tr_rays]], self.novel_K.expand(
            self.num_rays-num_tr_rays, -1, -1)))
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:, 0, 2] + 0.5) / K[:, 0, 0],
                    (y - K[:, 1, 2] + 0.5) / K[:, 1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / \
            torch.linalg.norm(directions, dim=-1, keepdims=True)

        origins = torch.reshape(origins, (self.num_rays, 3))  # num_rays, 3
        viewdirs = torch.reshape(viewdirs, (self.num_rays, 3))  # num_rays, 3
        rgba = torch.reshape(
            rgba, (num_tr_rays, rgba.shape[1]))  # num_tr_rays, 4

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=image_id)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }

    def fetch_data_depth(self, fraction, index):
        '''fraction: Proportion of rays that has a 3d point'''

        num_rays_ds = int(self.num_rays//(1/fraction))
        num_rays_rgb = int(self.num_rays-num_rays_ds)

        # get image_id, x, y for rgb rays
        image_id = torch.randint(0, len(self.images), size=(
            num_rays_rgb,), device=self.images.device)
        x = torch.randint(0, self.WIDTH, size=(
            num_rays_rgb,), device=self.images.device)
        y = torch.randint(0, self.HEIGHT, size=(
            num_rays_rgb,), device=self.images.device)

        # append image_id, x, y for depth rays to the front
        depth_id = torch.randint(0, self.depths.shape[0], size=(
            num_rays_ds,), device=self.images.device)
        _, im_id, xd, yd, depths, errs = self.depths[depth_id].T
        image_id = torch.cat((im_id.type(torch.long), image_id), 0)
        x = torch.cat((xd.type(torch.long), x), 0)
        y = torch.cat((yd.type(torch.long), y), 0)
        depth_err = torch.vstack((depths, errs)).T

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.c2w[image_id]  # (num_rays, 3, 4)
        K = self.K[image_id]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:, 0, 2] + 0.5) / K[:, 0, 0],
                    (y - K[:, 1, 2] + 0.5) / K[:, 1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / \
            torch.linalg.norm(directions, dim=-1, keepdims=True)

        origins = torch.reshape(origins, (self.num_rays, 3))
        viewdirs = torch.reshape(viewdirs, (self.num_rays, 3))
        rgba = torch.reshape(rgba, (self.num_rays, rgba.shape[1]))

        rays = Rays(origins=origins, viewdirs=viewdirs, image_id=image_id)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "depth_err": depth_err,  # [num_rays, 2]
        }

    def get_depth_rays(self):
        num_rays = self.num_rays
        idx = torch.randint(0, self.depths.shape[1], size=(
            num_rays,), device=self.images.device)

        rand_depths = self.depths[:, idx]
        image_id, x, y, depths = np.vsplit(rand_depths, 4)

        image_id = image_id.reshape([num_rays]).to(torch.long)
        x = x.reshape([num_rays]).to(torch.long)
        y = y.reshape([num_rays]).to(torch.long)

        c2w = self.c2w[image_id]  # (num_rays, 3, 4)
        K = self.K[image_id]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:, 0, 2] + 0.5) / K[:, 0, 0],
                    (y - K[:, 1, 2] + 0.5) / K[:, 1, 1] *
                    (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        origins = torch.reshape(origins, (num_rays, 3))
        viewdirs = torch.reshape(viewdirs, (num_rays, 3))
        depths = torch.reshape(depths, (num_rays, 1))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.c2w.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.c2w.device)

        return {
            "depths": depths,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": color_bkgd
        }

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.images.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.images.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.images.device)

        if rgba.shape[-1] == 4:
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
            # pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        else:
            pixels = rgba
            alpha = torch.ones_like(pixels[:, 0], device=self.device)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "mask": alpha,
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }
