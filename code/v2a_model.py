import pytorch_lightning as pl
import torch.optim as optim
from lib.model.v2a_spatial import V2A ############################## # v2a, v2a_clip, v2a_spatial, v2a_spatial_ienc
from lib.model.body_model_params import BodyModelParams
from lib.model.deformer import SMPLDeformer
import cv2
import torch
from lib.model.loss import Loss
import hydra
import os
import numpy as np
from lib.utils.meshing import generate_mesh
from kaolin.ops.mesh import index_vertices_by_faces
import trimesh
from lib.model.deformer import skinning
from lib.utils import utils
import pdb
class V2AModel(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()
        ########################################
        self.validation_step_outputs = []
        ########################################

        self.opt = opt
        
        num_training_frames = opt.dataset.metainfo.end_frame - opt.dataset.metainfo.start_frame

        ############################################################################
        # num_training_frames -= len(self.opt.dataset.test['image_id'])
        ############################################################################

        self.betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.metainfo.data_dir, 'mean_shape.npy')
        self.gender = opt.dataset.metainfo.gender
        self.model = V2A(opt.model, self.betas_path, self.gender, num_training_frames, is_training=True)
        # self.get_clip_img_enc = Get_Clip_Enc()
        self.start_frame = opt.dataset.metainfo.start_frame
        self.end_frame = opt.dataset.metainfo.end_frame
        self.training_modules = ["model"]

        ############################################################################
        self.training_indices = list(range(self.start_frame, self.end_frame))
        # tmp = list(range(self.start_frame, self.end_frame))
        # self.training_indices = [item for item in tmp if item not in self.opt.dataset.train['except_image_id']]
        ############################################################################

        self.body_model_params = BodyModelParams(num_training_frames, model_type='smpl')
        self.load_body_model_params()
        optim_params = self.body_model_params.param_names
        for param_name in optim_params:
            self.body_model_params.set_requires_grad(param_name, requires_grad=True)
        self.training_modules += ['body_model_params']
        
        self.loss = Loss(opt.model.loss)
        
    def load_body_model_params(self):
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}
        data_root = os.path.join('../data', self.opt.dataset.metainfo.data_dir)
        data_root = hydra.utils.to_absolute_path(data_root)

        body_model_params['betas'] = torch.tensor(np.load(os.path.join(data_root, 'mean_shape.npy'))[None], dtype=torch.float32)
        body_model_params['global_orient'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices][:, :3], dtype=torch.float32)
        body_model_params['body_pose'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices] [:, 3:], dtype=torch.float32)
        body_model_params['transl'] = torch.tensor(np.load(os.path.join(data_root, 'normalize_trans.npy'))[self.training_indices], dtype=torch.float32)

        for param_name in body_model_params.keys():
            self.body_model_params.init_parameters(param_name, body_model_params[param_name], requires_grad=False) 

    def configure_optimizers(self):
        params = [{'params': self.model.parameters(), 'lr':self.opt.model.learning_rate}]
        params.append({'params': self.body_model_params.parameters(), 'lr':self.opt.model.learning_rate*0.1})
        self.optimizer = optim.Adam(params, lr=self.opt.model.learning_rate, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.1)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch):
        inputs, targets = batch

        batch_idx = inputs["idx"]

        body_model_params = self.body_model_params(batch_idx)
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        inputs['current_epoch'] = self.current_epoch
        # pdb.set_trace()
        model_outputs = self.model(inputs)

        #############################################################################
        # clip_img_feature = self.get_clip_img_enc(targets)
        # targets['clip_img_feature'] = clip_img_feature # torch.Size([1, 512])
        #############################################################################

        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item(), prog_bar=True, on_step=True)
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)
        return loss_output["loss"]

    def on_training_epoch_end(self, outputs) -> None:        
        # Canonical mesh update every 20 epochs
        if self.current_epoch != 0 and self.current_epoch % 20 == 0:
            cond = {'smpl': torch.zeros(1, 69).float().cuda()}
            mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=2)
            self.model.mesh_v_cano = torch.tensor(mesh_canonical.vertices[None], device = self.model.smpl_v_cano.device).float()
            self.model.mesh_f_cano = torch.tensor(mesh_canonical.faces.astype(np.int64), device=self.model.smpl_v_cano.device)
            self.model.mesh_face_vertices = index_vertices_by_faces(self.model.mesh_v_cano, self.model.mesh_f_cano)
        return super().on_training_epoch_end(outputs)

    def query_oc(self, x, cond):
        
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.implicit_network(x, cond)[:,:,0].reshape(-1,1)
        return {'sdf':mnfld_pred}

    def query_wc(self, x):
        
        x = x.reshape(-1, 3)
        w = self.model.deformer.query_weights(x)
    
        return w

    def query_od(self, x, cond, smpl_tfs, smpl_verts):
        
        x = x.reshape(-1, 3)
        x_c, _ = self.model.deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
        output = self.model.implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        
        return {'sdf': sdf}

    def get_deformed_mesh_fast_mode(self, verts, smpl_tfs):
        verts = torch.tensor(verts).cuda().float()
        weights = self.model.deformer.query_weights(verts)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def validation_step(self, batch, *args, **kwargs):

        output = {}
        inputs, targets = batch
        inputs['current_epoch'] = self.current_epoch
        self.model.eval()

        body_model_params = self.body_model_params(inputs['image_id'])
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        cond = {'smpl': inputs["smpl_pose"][:, 3:]/np.pi}
        mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=3)
        
        mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces)
        
        output.update({
            'canonical_mesh':mesh_canonical
        })

        split = utils.split_input(inputs, targets["total_pixels"][0], n_pixels=min(targets['pixel_per_batch'], targets["img_size"][0] * targets["img_size"][1]))

        res = []
        for s in split:

            out = self.model(s)

            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v

            res.append({
                'rgb_values': out['rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                'fg_rgb_values': out['fg_rgb_values'].detach(),
                'fg_clip_sdf_values': out['fg_clip_sdf_values'].detach(),
            })
        batch_size = targets['rgb'].shape[0]
        
        

        model_outputs = utils.merge_output(res, targets["total_pixels"][0], batch_size) # targets["total_pixels"] 921600==1280x720

        ##############################################################################################################
        fg_clip_sdf_values = ((torch.nn.functional.sigmoid(model_outputs['fg_clip_sdf_values']*10))-0.5)*2
        ##############################################################################################################

        output.update({
            "rgb_values": model_outputs["rgb_values"].detach().clone(),
            "normal_values": model_outputs["normal_values"].detach().clone(),
            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
            "fg_clip_sdf_values": fg_clip_sdf_values.detach().clone(),
            **targets,
        })
        self.validation_step_outputs.append(output)
            
        return output

    def validation_step_end(self, batch_parts):
        return batch_parts

    # pytorch-lightning 2.0.6으로 update함. validation_epoch_end()대신 on_validation_epoch_end() 사용
    # 파라미터 outputs 없앰
    # 대신, init에서 정의하고, validataion_step에서 추가한 self.validation_step_outputs 이용함
    # https://github.com/Lightning-AI/lightning/discussions/17182
    # 
    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        img_size = outputs[0]["img_size"]

        rgb_pred = torch.cat([output["rgb_values"] for output in outputs], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([output["fg_rgb_values"] for output in outputs], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([output["normal_values"] for output in outputs], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([output["rgb"] for output in outputs], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        if 'normal' in outputs[0].keys():
            normal_gt = torch.cat([output["normal"] for output in outputs], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        os.makedirs("rendering", exist_ok=True)
        os.makedirs("normal", exist_ok=True)
        os.makedirs('fg_rendering', exist_ok=True)

        canonical_mesh = outputs[0]['canonical_mesh']
        canonical_mesh.export(f"rendering/{self.current_epoch}.ply")


        ##############################################################################################################
        fg_clip_sdf_values_pred = torch.cat([output["fg_clip_sdf_values"] for output in outputs], dim=0)
        fg_clip_sdf_values_pred = fg_clip_sdf_values_pred.reshape(*img_size, -1).cpu().numpy()
        fg_clip_sdf_values_pred = (fg_clip_sdf_values_pred*255).astype(np.uint8)
        os.makedirs(f'clip_sdf/{self.current_epoch}', exist_ok=True)
        for i in range(fg_clip_sdf_values_pred.shape[-1]):
            cv2.imwrite(f"clip_sdf/{self.current_epoch}/{i}.png", fg_clip_sdf_values_pred[:, :, i])
        ##############################################################################################################

        try:
            # (1440, 1280, 6)이 나오는 이유는? self.validation_step_outputs 리스트 초기화 안해서
            #원래는 (1440, 1280, 3)이 나와야함
            cv2.imwrite(f"rendering/{self.current_epoch}.png", rgb[:, :, ::-1]) 
            cv2.imwrite(f"normal/{self.current_epoch}.png", normal[:, :, ::-1])
            cv2.imwrite(f"fg_rendering/{self.current_epoch}.png", fg_rgb[:, :, ::-1])
        except Exception as e:
            print(e)
            pdb.set_trace()
        self.validation_step_outputs = []


    ##################################################3
    ##################################################3
    ## test 할 때 이 함수 이름 training_step으로 바꾸고 원래 training step은 주석처리
    def testing_step(self, batch, *args, **kwargs):
        # pdb.set_trace()


        # torch.set_grad_enabled(True)
        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                       1) // pixel_per_batch
        results = []

        ############################3
        batch_idx = inputs["idx"]

        body_model_params = self.body_model_params(batch_idx)
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        inputs['current_epoch'] = self.current_epoch
        ############################3
        # pdb.set_trace()





        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=1)

        body_model_params = self.body_model_params(inputs['idx'])
        # pdb.set_trace()
        smpl_shape = body_model_params['betas'] if body_model_params['betas'].dim() == 2 else body_model_params['betas'].unsqueeze(0)
        smpl_trans = body_model_params['transl']
        smpl_pose = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)

        smpl_outputs = self.model.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs['smpl_tfs']
        cond = {'smpl': smpl_pose[:, 3:]/np.pi}

        mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=4)
        self.model.deformer = SMPLDeformer(betas=np.load(self.betas_path), gender=self.gender, K=7)
        verts_deformed = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, smpl_tfs)
        mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)

        os.makedirs("test_mask", exist_ok=True)
        os.makedirs("test_rendering", exist_ok=True)
        os.makedirs("test_fg_rendering", exist_ok=True)
        os.makedirs("test_normal", exist_ok=True)
        os.makedirs("test_mesh", exist_ok=True)
        
        mesh_canonical.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_canonical.ply")
        mesh_deformed.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_deformed.ply")
        self.model.deformer = SMPLDeformer(betas=np.load(self.betas_path), gender=self.gender)
        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "intrinsics": inputs['intrinsics'],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:, 4:76],
                            "smpl_shape": inputs["smpl_params"][:, 76:],
                            "smpl_trans": inputs["smpl_params"][:, 1:4],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None,
                            "current_epoch": inputs['current_epoch'] }

            body_model_params = self.body_model_params(inputs['idx'])

            batch_inputs.update({'smpl_pose': torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)})
            batch_inputs.update({'smpl_shape': body_model_params['betas']})
            batch_inputs.update({'smpl_trans': body_model_params['transl']})

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            
            results.append({"rgb_values":model_outputs["rgb_values"].detach().clone(), 
                            "fg_rgb_values":model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            **batch_targets})         

        # pdb.set_trace()
        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = pred_mask.reshape(*img_size, -1)

        if results[0]['rgb'] is not None:
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        else:
            rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()
        if 'normal' in results[0].keys():
            normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()
        
        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        cv2.imwrite(f"test_mask/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
        cv2.imwrite(f"test_rendering/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])